
import copy
import torch
import torch.nn as nn
from replay_buffer import ReplayBuffer
import torch.nn.functional as F
from .feature import pca_weights
import torch.optim as optim
criterion = nn.CrossEntropyLoss()

class Worker_Vision:
    def __init__(self, model, rank, optimizer, scheduler,
                 train_loader, device):       
        self.model = model
        self.rank = rank
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        # self.train_loader_iter = train_loader.__iter__()
        self.device = device


    def update_iter(self):
        self.train_loader_iter = self.train_loader.__iter__()

    def step(self):
        self.model.train()

        batch = self.train_loader_iter.__next__()
        self.data, self.target = batch[0].to(self.device), batch[1].to(self.device)
        output = self.model(self.data)
        loss = criterion(output, self.target)
        self.optimizer.zero_grad()
        loss.backward()

    def refresh_bn(self):
        self.model.train()

        batch = self.train_loader_iter.__next__()
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        self.model(data)
        # loss = criterion(output, target)
        # self.optimizer.zero_grad()
        # loss.backward()

    def step_csgd(self):
        self.model.train()

        batch = self.train_loader_iter.next()
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        output = self.model(data)
        loss = criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()

        grad_dict = {}
        for name, param in self.model.named_parameters():
            grad_dict[name] = param.grad.data

        return grad_dict
    
    def get_accuracy(self, model):
        model.eval()
        output = model(self.data)
        _, predicted = torch.max(output.data, 1)
        total_samples = self.target.size(0)
        total_correct = (predicted == self.target).sum().item()
        accuracy = total_correct / total_samples
        return accuracy
        

    def update_grad(self):
        self.optimizer.step()
        self.scheduler.step()

    def scheduler_step(self):
        self.scheduler.step()

# 传入一个定义好的network
class DQNAgent(Worker_Vision):
    """DQN Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
    """

    def __init__(
        self, 
        model, 
        value_model,
        rank, 
        optimizer, 
        scheduler,
        train_loader, 
        args,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
        memory_size: int = 10000,
        batch_size: int = 10,
        target_update: int = 10,
        epsilon_decay: float = 1/2000,
        seed: int = 6666
        ):
        super().__init__(model, rank, optimizer, scheduler,
                 train_loader, args.device)
        
        self.state_size = args.state_size
        self.memory = ReplayBuffer(self.state_size, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        self.clients_number = args.size # clients总数
        # pac 系数
        self.n_components = args.n_components
        # device: cpu / gpu
        self.device = args.device
        # print(self.device)
        self.sample = args.sample

        # networks: dqn, dqn_target
        # self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn = value_model.to(self.device)
        # self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = value_model.to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # dqn网络的optimizer
        self.dqn_optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

        
        self.update_cnt = 0

    def feature(self, n_components, model):
        weights_pca = pca_weights(n_components=n_components, weights=model.state_dict())
        # weights_pca = torch.from_numpy(weights_pca)
        return weights_pca


    def select_action_sample(self):
        # 增加采样数量
        if self.sample < 0:
            print('invalid sample')
        else:
            # 假设 self.dqn 是一个 PyTorch 模型，self.state 是输入状态
            # 获取网络输出
            output = self.dqn(self.state.to(self.device)) # output shape [size]
            num_samples = self.sample
            # 使用 multinomial 函数根据输出概率随机选择下标
            # 注意：multinomial 函数返回的是下标，而不是概率值
            # replacement=False 表示不重复选择
            # dim=1 表示在输出张量的最后一个维度上进行选择
            selected_indices = torch.multinomial(output.softmax(dim=-1), num_samples, replacement=False)
            # selected_indices shape [sample] 
            # 使用选择的下标来获取对应的输出值 selected_indices 和 output shape 相同 都是一维
            selected_outputs = output.gather(0, selected_indices)

            self.transition_sample = list()
            for i in range(0, num_samples):
                self.transition_sample.append([self.state, selected_indices[i].item()])
                merge_model = self.act(self.model, selected_indices[i], self.worker_list_model)
                old_accuracy = self.get_accuracy(self.model)
                new_accuracy = self.get_accuracy(merge_model)
                reward = new_accuracy - old_accuracy
                next_state = self.feature(self.n_components, merge_model)
                done = 0
                if not self.is_test:
                    self.transition_sample[i] += [reward, next_state, done]
                self.memory.store(*self.transition_sample[i])
            # 现在 selected_indices 包含了选择的下标，selected_outputs 包含了对应的输出值
            # self.store_buffer_sample(*self.transition_sample[i])

# 这个函数暂时用不上            
    '''  
    def store_buffer_sample(self):
        for i in range(0, len(self.transition_sample)):
            # 这里的三个参数的计算到后面再优化
            done = 0
            next_state = self.state
            reward = new_acc - old_acc
            if not self.is_test:
                .transition_sample[i] += [reward, next_state, done]
                self.memory.store(*self.transition_sample[i])
    '''


    def select_action(self) -> int:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > torch.rand(1).item():
            # selected_action = self.env.action_space.sample()
            action_space = torch.tensor([i for i in range(0, self.clients_number)])
            selected_action = action_space[torch.randint(low=0, high=len(action_space), size=(1,))].item()
        else:
            selected_action = self.dqn(
                self.state.to(self.device)
            ).argmax().item()
            # selected_action = selected_action.detach().cpu().numpy()
        
        # 在这里先存好状态和动作
        if not self.is_test:
            self.transition = [self.state, selected_action]
        
        return selected_action

    def store_buffer(self, old_acc, new_acc):
        """Take an action and return the response of the env."""
        # next_state, reward, terminated, truncated, _ = self.env.step(action)
        # 这里设定next state 和state 相同 ， done设置为False，reward 用准确率的变化来计算
        done = 0
        next_state = self.feature(self.n_components, self.model)
        reward = new_acc - old_acc

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
    
    # 更新策略网络
    def update_dqn(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)

        self.dqn_optimizer.zero_grad()
        loss.backward()
        self.dqn_optimizer.step()

        return loss.item()
    
    # 做出行动
    def act(self, model, action, worker_list):
        model = copy.deepcopy(model)
        for name, param in model.named_parameters():
            choose_worker = worker_list[action]
            param.data += choose_worker.state_dict()[name].data
            param.data /= 2
        return model

    def get_workerlist(self, worker_list):
        self.worker_list_model = worker_list

    # 这个方法用于在每个step里面模型融合
    def step_updatemodel(self,worker_list):
        action = self.select_action()
        self.model = self.act(self.model, action, worker_list)
    
    def train_step_dqn(self):
        # action = self.select_action(self.state)
        # next_state, reward, done = self.step(action)
        # next_state = self.state
        
        self.state = self.feature(self.n_components, self.model)

        # 在选择action并作出action前先判断是否要sample
        if self.sample > 0:
            self.select_action_sample()
        
        # 这一步的作用是选择action并作出action
        self.step_updatemodel(self.worker_list_model)
        
        # 思考done的含义？
        # if episode ends
        # if done:
            # state, _ = self.env.reset(seed=self.seed)
            # scores.append(score)
            # score = 0
        # 在这里更新策略函数
        # if training is ready
        if len(self.memory) >= self.batch_size:
            loss = self.update_dqn()
            # losses.append(loss)
            self.update_cnt += 1
            
            # linearly decrease epsilon
            self.epsilon = max(
                self.min_epsilon, self.epsilon - (
                    self.max_epsilon - self.min_epsilon
                ) * self.epsilon_decay
            )
            # epsilons.append(self.epsilon)
            
            # if hard update is needed
            if self.update_cnt % self.target_update == 0:
                self._target_hard_update()


    '''
    def train(self, num_frames: int):
        """Train the agent."""
        self.is_test = False
        
        state, _ = self.env.reset(seed=self.seed)
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # if episode ends
            # if done:
                # state, _ = self.env.reset(seed=self.seed)
                # scores.append(score)
                # score = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1
                
                # linearly decrease epsilon
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
                epsilons.append(self.epsilon)
                
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # plotting
            # if frame_idx % plotting_interval == 0:
                # self._plot(frame_idx, scores, losses, epsilons)
                
        # self.env.close()
        '''
                
    def test(self, video_folder: str) -> None:
        """Test the agent."""
        self.is_test = True
        
        
        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        print("score: ", score)
        self.env.close()
        
        # reset
        # self.env = naive_env

    def _compute_dqn_loss(self, samples) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        action = action.long()
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(
            next_state
        ).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                


from torch.cuda.amp.grad_scaler import GradScaler
scaler = GradScaler()
class Worker_Vision_AMP:
    def __init__(self, model, rank, optimizer, scheduler,
                 train_loader, device):       
        self.model = model
        self.rank = rank
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        # self.train_loader_iter = train_loader.__iter__()
        self.device = device


    def update_iter(self):
        self.train_loader_iter = self.train_loader.__iter__()

    def step(self):
        self.model.train()

        batch = self.train_loader_iter.next()
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        with torch.cuda.amp.autocast(enabled=True,dtype=torch.float16):
            output = self.model(data)
            loss = criterion(output, target)
        self.optimizer.zero_grad()
        scaler.scale(loss).backward()
        # loss.backward()

    def step_csgd(self):
        self.model.train()

        batch = self.train_loader_iter.next()
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        with torch.cuda.amp.autocast(enabled=True,dtype=torch.float16):
            output = self.model(data)
            loss = criterion(output, target)
        self.optimizer.zero_grad()
        scaler.scale(loss).backward()
        # loss.backward()

        grad_dict = {}
        for name, param in self.model.named_parameters():
            grad_dict[name] = param.grad.data

        return grad_dict

    def update_grad(self):
        # self.optimizer.step()
        scaler.step(self.optimizer)
        scaler.update()
        self.scheduler.step()

    def scheduler_step(self):
        self.scheduler.step()
