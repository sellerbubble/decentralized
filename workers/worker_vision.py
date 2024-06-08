
import torch
import torch.nn as nn
from replay_buffer import ReplayBuffer
import torch.nn.functional as F
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
    
    def get_accuracy(self):
        self.model.eval()
        output = self.model(self.data)
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
        device,
        node_number,
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
                 train_loader, device)
        
        # obs dim 设置为 1 
        self.memory = ReplayBuffer(1, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        self.node_number = node_number
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks: dqn, dqn_target
        # self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn = value_model.to(self.device)
        # self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = value_model.to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer 使用原本的优化器
        # self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

        # 先将state 简单化
        self.state = torch.tensor(self.node_number+1).to(self.device)
        self.state = self.state.to(torch.float32)

        self.update_cnt = 0

    def select_action(self) -> int:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > torch.rand(1).item():
            # selected_action = self.env.action_space.sample()
            action_space = torch.tensor([i for i in range(0, self.node_number )])
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
        next_state = self.state
        reward = new_acc - old_acc

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
    

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    # 这个方法用于在每个step里面模型融合
    def step_update(self,worker_list):
        action = self.select_action()
        for name, param in self.model.named_parameters():
            choose_worker = worker_list[action]
            param.data += choose_worker.model.state_dict()[name].data
            param.data /= 2
    
    def train_step_dqn(self, worker_list):
        # action = self.select_action(self.state)
        # next_state, reward, done = self.step(action)
        # next_state = self.state

        self.step_update(worker_list)
        

        # if episode ends
        # if done:
            # state, _ = self.env.reset(seed=self.seed)
            # scores.append(score)
            # score = 0

        # if training is ready
        if len(self.memory) >= self.batch_size:
            loss = self.update_model()
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
