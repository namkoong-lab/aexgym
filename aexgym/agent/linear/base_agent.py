import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import Optional
from aexgym.model.base_model import BaseModel 

class LinearAgent(nn.Module): 
    """
    Superclass for contextual linear bandit agent. Relies on an fake model 
    environment to provide a feature representation and to rollout 
    simulated trajectories. 
    """
    def __init__(self, 
                 model: BaseModel,
                 name: str,
                 **kwargs):
        super().__init__()
        self.model = model
        self.name = name
        self.use_precision = model.use_precision 

    def train_agent(*args, **kwargs):
        pass
    
    def forward(self, 
                beta: Tensor, 
                sigma: Tensor, 
                contexts: Optional[Tensor] = None):
        
        """ Outputs a probability distribution over arm choices given a context.
        Args:

            contexts (tensor): (batch_size x context_len) tensor of contexts
        """
        pass 

    def fantasize(self, 
                beta: Tensor, 
                contexts: Tensor, 
                action_contexts: Tensor):
        """pick separate best arm for each context"""
        phis = self.model.features_all_arms(contexts, action_contexts)
        rewards = torch.einsum('nkc,cd->nkd', phis, beta)
        return rewards



    

class LinearUniform(LinearAgent):
    def __init__(self, model, name, **kwargs):
        super().__init__(model, name)

    def forward(self, 
                beta, 
                sigma, 
                contexts=None, 
                action_contexts = None, 
                objective=None, 
                costs=None):
        batch_size = contexts.shape[0]
        return torch.ones((batch_size, self.model.n_arms), device=contexts.device) / self.model.n_arms



