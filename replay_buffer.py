from typing import List
import numpy as np
import torch
# from typing import Dict, List, Tuple

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, 
                 obs_dim: int, # 还不清楚state的具体定义，因此这里无法传入observation state 的dim.必须传入 因为后面要用张量作为索引
                 size: int, batch_size: int = 32):
        self.obs_buf = torch.zeros([size, obs_dim], dtype=torch.float32)
        self.next_obs_buf = torch.zeros([size, obs_dim], dtype=torch.float32)
        
        self.acts_buf = torch.zeros(size, dtype=torch.float32)
        self.rews_buf = torch.zeros(size, dtype=torch.float32)
        self.done_buf = torch.zeros(size, dtype=torch.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs,
        act: torch.tensor,
        rew: float, 
        next_obs, 
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # def sample_batch(self) -> Dict[str, np.ndarray]:
    def sample_batch(self):
        # idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        idxs = torch.randperm(self.size)[:self.batch_size]
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size
    
