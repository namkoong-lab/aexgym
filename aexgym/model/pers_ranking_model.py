import torch
from typing import Optional
from torch import Tensor
from aexgym.model import BaseLinearModel

class PersonalizedRankingModel(BaseLinearModel):
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
        user_contexts, item_contexts = contexts 
        num_items, action_contexts = action_contexts
        batch_size = user_contexts.shape[0]
        context_len = user_contexts.shape[1]
        rankers = action_contexts[actions]
        contexts = torch.einsum('nf, nif -> nif', user_contexts, item_contexts)
        scores = torch.einsum('nif, nf -> ni', contexts, rankers)
        max_indices = torch.topk(scores, num_items, largest=True, sorted=False).indices
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(batch_size, num_items)
        return contexts[batch_indices, max_indices]
    
    def features_all_arms(self, contexts: Tensor, action_contexts: Tensor):
        user_contexts, item_contexts = contexts 
        batch_size = user_contexts.shape[0]
        features = []
        for i in range(self.n_arms):
            actions = torch.tensor([i] * batch_size)
            features.append(self.feature_map(actions, contexts, action_contexts).unsqueeze(1))
        return torch.cat(features, dim=1)