import torch
from torch import nn
from torch.nn import functional as F

class RegAgent(nn.Module): 

    def __init__(self, MDP, name):
        super().__init__()
        """ Initialize contextual bandit agent. 
        
        Args:
            env (obj): environment object
            name (string): name of  agent
        
        """
        self.MDP = MDP
        self.name = name
        self.use_precision = MDP.use_precision 

    def train_agent(*args):
        pass
    
    def forward(self, beta, sigma, contexts = None):
        
        """ Pick actions for given contexts.

        Args:
            contexts (tensor): (batch_size x context_len) tensor of contexts
        """
        pass 

    def exploit(self, beta, contexts):
        """pick separate best arm for each context"""
        batch_size = contexts.shape[0]
        max_idx = torch.argmax(beta)
        return (max_idx*torch.ones(batch_size)).int()
    
    def get_best_arm(self, beta, *args):
        """picks one best arm to deploy across all contexts"""
        
        max_idx = torch.argmax(beta)
        return max_idx.item()



class Uniform(RegAgent):
    def __init__(self, MDP, name):
        super().__init__(MDP, name)

    def forward(self, beta, sigma, contexts=None):
        if contexts is None:
            return torch.ones(self.MDP.n_arms, device = beta.device) / self.MDP.n_arms
        else:
            batch_size = contexts.shape[0]
            return torch.ones((batch_size, self.MDP.n_arms), device=contexts.device) / self.MDP.n_arms



