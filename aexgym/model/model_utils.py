import torch 
from torch import Tensor
from typing import Optional
from torch.nn import functional as F

def update_linear_posterior(beta: Tensor, 
                            sigma: Tensor, 
                            rewards: Tensor,
                            features: Tensor, 
                            s2: Tensor,  
                            idx: Tensor,
                            use_precision: Optional[bool] = False,
                            n_objs: Optional[int] = 1,
                            standardize:Optional[bool]=True):
        
    """ Update posterior using linear updates.

    Args:
        rewards (tensor): 1D tensor of reward sequence
        features (tensor): (batch_size x feature_len) tensor of corresponding features
    """
    batch_size_size = features.shape[0]
    s2_mean = torch.mean(s2, dim=0)
    s2 = s2[idx]
    standardize=False
    if standardize:
        rewards = (rewards - torch.mean(rewards, dim=0, keepdim=True)) / (s2_mean.to(rewards.device) ** 0.5)
        s2 = s2 / s2_mean
    XTX = torch.matmul(features.T, features)
    XR = torch.matmul(features.T, rewards)
    #update 
    if not use_precision:
        get_sigma = lambda sigma, s2: torch.linalg.inv(torch.linalg.inv(sigma) + XTX / (s2))
        get_beta = lambda beta, sigma, post_sigma, s2, XR: post_sigma @ ( XR / (s2) + torch.linalg.inv(sigma) @ beta)

    elif use_precision:
        get_sigma = lambda sigma, s2: sigma + XTX / (s2)
        get_beta = lambda beta, sigma, post_sigma, s2, XR: torch.linalg.inv(post_sigma) @ (XR / (s2) + sigma @ beta)

    post_sigma = torch.stack([get_sigma(sigma[:, :, i], s2[i]) for i in range(n_objs)], dim=2)
    post_beta = torch.stack([get_beta(beta[:, i], sigma[:, :, i], post_sigma[:, :, i], s2[i], XR[:, i]) for i in range(n_objs)], dim = 1)
    return post_beta, post_sigma

def update_reg_posterior(mu: Tensor, 
                         var: Tensor, 
                         actions: Tensor, 
                         rewards: Tensor, 
                         s2: float, 
                         use_precision: Optional[bool] = False):
        
    """ Update posterior using Gaussian Updates.

    Args: 
        rewards (tensor): 1D tensor of reward sequence
        actions (tensor): 1D tensor of corresponding actions

    """
    one_hot_matrix = F.one_hot(actions, len(mu))


    counts = torch.sum(one_hot_matrix, axis=0)
    rewards = torch.sum(rewards.unsqueeze(1)*one_hot_matrix, axis=0)
    
    post_var = s2*var/(s2 + counts*var)
    mu = post_var*(mu / var + (rewards/s2))
    
    return mu, var