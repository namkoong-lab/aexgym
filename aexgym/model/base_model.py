"""Base environment class for simulating contextual bandit problems"""

import abc 
from aexgym.model.model_utils import update_linear_posterior, update_reg_posterior
from typing import Callable, Dict, List, Optional, Tuple, Union
from torch import Tensor
import torch

class BaseModel:
    """
    Base class for non-contextual Model environment. Provides state 
    transitions using Gaussian updates to the posterior, and can 
    simulate rollouts for any policy. 
    """
    def __init__(self,
                 beta_0: Tensor,
                 sigma_0: Tensor, 
                 n_arms: int, 
                 s2: float,  
                 use_precision: Optional[bool] = False):
        
        self.beta_0 = beta_0 
        self.sigma_0 = sigma_0 
        self.n_arms = n_arms
        self.s2 = s2
        self.use_precision = use_precision 
    
    def update_posterior(self, mu: Tensor, var: Tensor, rewards: Tensor, actions: Tensor, 
                         *args):
        return update_reg_posterior(mu, var, actions, rewards, self.s2, self.use_precision)
        
    def reset(self):
        #for convenience, the mean and variance are labeled as beta and sigma
        return self.beta_0, self.sigma_0 


class BaseLinearModel:
    """
    Base class for linear contextual Model environment. Provides 
    linear posterior update and feature map that maps contexts/action 
    pairs to features. 
    
    """
    def __init__(self, 
                 beta_0: Tensor,
                 sigma_0: Tensor,
                 n_arms: int,
                 s2: Tensor,  
                 use_precision: Optional[bool] = False,
                 n_objs: Optional[int] = 1,
                 standardize: Optional[bool]=True):
        
        self.beta_0 = beta_0 
        self.sigma_0 = sigma_0 
        self.n_arms = n_arms
        self.s2 = s2
        self.use_precision = use_precision 
        self.n_objs = n_objs
        self.standardize = standardize
    
    @abc.abstractmethod
    def feature_map(self, actions: Tensor, contexts: Tensor):
        
        """ Creates feature vectors phi(x,a) for list of arms and contexts

        Args: 
            actions (tensor): 1D tensor of arm indices
            contexts (tensor): 2D (batch_size x context_len) tensor of contexts

        Returns:
            phi (tensor): 2D (batch_size x feature_len) tensor of feature vectors
        """
        pass
    
    def features_all_arms(self, contexts: Tensor):
        
        """ Prepares to find best arm for each context by 
         stacking phi(x,a) for all arms for given contexts. 
         Below is a general implementation, adjust this method if there 
         are more efficient implementations. 

         Args:
            contexts (tensor): 2D (batch_size x context_len) tensor of contexts

        Returns:
            phi (tensor): 3D (batch_size x n_arms x feature_len) tensor of feature vectors
        """
 
        batch_size = contexts.shape[0]
        list_of_features = [self.feature_map(i*torch.ones(batch_size, dtype=int, device = contexts.device), contexts).unsqueeze(1) for i in range(self.n_arms)]
        return torch.cat(list_of_features, dim=1)

    def update_posterior(self, 
                         beta:Tensor, 
                         sigma: Tensor, 
                         rewards: Tensor,
                         features: Tensor,
                         idx: int):
        return update_linear_posterior(beta, sigma, rewards,  features, self.s2, idx, self.use_precision, self.n_objs, standardize=self.standardize) 
        
    def reset(self):
        return self.beta_0, self.sigma_0 
    
