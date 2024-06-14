import torch 
from torch import Tensor
from typing import Optional, List
from aexgym.model.base_model import BaseLinearModel

class PersonalizedLinearModel(BaseLinearModel):
    """
    Assumes the true reward model is r(x,a) = \beta_a^T x. Models personalization settings 
    where arm coefficients do not depend on the context. 
    """
    
    def __init__(self, 
                 beta_0: Tensor, 
                 sigma_0: Tensor, 
                 n_arms: int,  
                 s2: Tensor,
                 use_precision: Optional[bool] =  False,
                 n_objs: Optional[int] = 1,
                 standardize: Optional[bool] = True): 
        
        super().__init__(beta_0, sigma_0, n_arms, s2, use_precision, standardize)
        
        context_len = int(len(beta_0) / n_arms)
        feature_len = len(beta_0)
        #create context matrix (for parallelized sampling)
        self.context_matrix = torch.zeros((n_arms, feature_len))
        for action, row in zip(torch.arange(n_arms), self.context_matrix):
            row[action*context_len:(action+1)*context_len] = torch.ones(context_len)

    def feature_map(self, 
                    actions: Tensor, 
                    contexts: Tensor, 
                    action_contexts: Tensor):
        """
        Maps an action and context to a feature as follows: 
        - create a zero tensor with length context_len * n_arms 
        - replaces [context_len * a : context_len * (a+1) entries with beta_a
        - then \beta^\top \phi(x,a) = \beta_a^\top x
        """
        batch_size = contexts.shape[0]
        context_len = contexts.shape[1]
        matrix = torch.zeros((batch_size,self.n_arms*context_len), device=contexts.device)

        for action, row, context in zip(actions, matrix, contexts):
            row[context_len*action:context_len*(action+1)] = context

        return matrix
    
    def features_all_arms(self, contexts: Tensor, action_contexts: Tensor):
        context_matrix = self.context_matrix.to(contexts.device)
        batch_size = contexts.shape[0]
        arm_idx = torch.arange(self.n_arms)
        arms_array = torch.unsqueeze(context_matrix, 0).repeat(batch_size, 1, 1)
        context_array = contexts.repeat(1, self.n_arms)
        return torch.einsum('nk,njk->njk', context_array, arms_array)
    
class fixedPersonalizedModel(BaseLinearModel):
    """
    Assumes the true reward model is r(a) = \beta^T z_a, where z_a are the features 
    for arm a. Models personalization settings where there is one set of coefficients for all arms.  
    """
    def __init__(self, 
                 beta_0: Tensor, 
                 sigma_0: Tensor, 
                 n_arms: int,  
                 s2: Tensor, 
                 use_precision: Optional[bool] =  False,
                 n_objs: Optional[int] = 1,
                 standardize: Optional[bool] = True): 
        
        super().__init__(beta_0, sigma_0, n_arms, s2, use_precision, standardize)
        
    def feature_map(self, 
                    actions: Tensor, 
                    contexts: Tensor,
                    action_contexts: Tensor):
        """
        There are no contexts in this model. Contexts is a batch_size length tensor. 
        Takes in actions and returns the context for each action 

        Args:  
            actions (tensor): (batch_size, ) tensor of actions 
            contexts (tensor): (batch_size, context_len) tensor of contexts, here it is a dummy variable encoding 
            batch_size size. 
            action_contexts (tensor): (n_arms, action_context_len) tensor of contexts for each arm.

        Returns:
            (batch_size, action_context_len) tensor of contexts for each action. 
        """
        batch_size = contexts.shape[0]
        features = action_contexts[actions]
        return torch.cat([features, -features], dim=1)
    def features_all_arms(self, 
                          contexts: Tensor, 
                          action_contexts: Tensor): 
        """
        Returns the features for all arms for a given context. 
        """
        batch_size = contexts.shape[0]
        features = torch.cat([action_contexts, -action_contexts], dim=1)
        return features.unsqueeze(0).repeat(batch_size, 1, 1)
     

