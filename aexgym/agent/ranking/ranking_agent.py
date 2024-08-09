import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import Optional
from aexgym.model.base_model import BaseModel 
from aexgym.agent.linear.base_agent import LinearAgent

class RankingAgent(LinearAgent): 
    """
    Superclass for contextual linear bandit agent. Relies on an fake model 
    environment to provide a feature representation and to rollout 
    simulated trajectories. 
    """
    def __init__(self, 
                 model: BaseModel,
                 name: str,
                 **kwargs):
        super().__init__(model, name) 
        self.use_precision = model.use_precision

    def fantasize(self, 
                beta: Tensor, 
                contexts: Tensor, 
                action_contexts: Tensor):
        """pick separate best arm for each context"""
        phis = torch.sum(self.model.features_all_arms(contexts, action_contexts), dim=2)
        rewards = torch.einsum('nkc,cd->nkd', phis, beta)
        return rewards

class RankingUniform(RankingAgent):
    def __init__(self, model, name, **kwargs):
        super().__init__(model, name)

    def forward(self,
                beta, 
                sigma, 
                contexts=None, 
                action_contexts = None, 
                objective=None, 
                costs=None):
        user_contexts, item_contexts = contexts 
        batch_size = user_contexts.shape[0]
        return torch.ones((batch_size, self.model.n_arms), device=beta.device) / self.model.n_arms