from torch import Tensor
import torch 
from typing import Optional
from aexgym.env.base_env import BaseContextualEnv

class PennEnv(BaseContextualEnv):


    def __init__(self, df, batch_size, n_steps: int):
        super().__init__(n_steps)
        self.df = df 
        self.batch_size = batch_size
        
    def sample_state_contexts(self, i: Optional[int] = 0):
        return self.contexts
    
    def sample_eval_contexts(self, access: Optional[bool] =False):
        if access:
            self.contexts, self.rewards = self.bootstrap_contexts()
            return self.contexts
        else:
            return self.contexts

    def sample_action_contexts(self):
        return torch.tensor([0])
    
    def sample_arms(self, 
                    actions: Tensor, 
                    contexts: Tensor, 
                    action_contexts: Tensor, 
                    i: Optional[int]=0):
            
        return self.curr_rewards[torch.arange(self.curr_rewards.shape[0], dtype=int), actions].unsqueeze(1), contexts
        
    def get_true_rewards(self, 
                        contexts: Tensor, 
                        action_contexts: Tensor):
        return self.curr_rewards.unsqueeze(2)

    def step(self, 
             state_contexts: Tensor, 
             action_contexts: Tensor, 
             actions: Tensor):
        all_contexts, sampled_rewards, sampled_contexts, self.counter = super().step(state_contexts, action_contexts, actions)
        self.contexts, self.curr_rewards = self.bootstrap_contexts()
        return all_contexts, sampled_rewards, sampled_contexts, self.counter

    def bootstrap_contexts(self):
        samples = torch.tensor(self.df.sample(n=self.batch_size, replace=True).values)
        return samples[:, :-7], samples[:, -7:]

    def reset(self):
        self.contexts, self.curr_rewards = self.bootstrap_contexts()
        return super().reset()