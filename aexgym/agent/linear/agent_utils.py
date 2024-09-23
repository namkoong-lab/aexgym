import torch
from torch import nn
from torch.nn import functional as F
#from AES.policies.linear.msqrt import MPA_Lya
MPA_Lya = None

""" Policy for RHO, either tabular or neural network. """
class TabularPolicy(nn.Module):
    def __init__(self, contexts, k):
        super().__init__()
        self.k = k
        self.batch = contexts.shape[0]
        self.policy = nn.Parameter(torch.ones((self.batch, k)) / k)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, contexts):
        return self.softmax(self.policy)

class PolicyNet(nn.Module):
    def __init__(self, contexts, k, width=64):
        super().__init__()
        self.k = k
        context_size = contexts.shape[1]
        self.fc1 = nn.Linear(context_size, width)
        self.ln1 = nn.LayerNorm(normalized_shape=width)
        torch.nn.init.zeros_(self.fc1.weight)
        torch.nn.init.constant_(self.fc1.bias.data, 0.01)
        self.fc2 = nn.Linear(width, k)
        torch.nn.init.zeros_(self.fc2.weight)
        torch.nn.init.constant_(self.fc2.bias.data, 0.01)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, context):
        x = F.relu(self.ln1(self.fc1(context)))
        x = self.softmax(self.fc2(x))
        return x
    

def LinearQFn(
    beta, 
    cov, 
    num_zs, 
    features_all_arms, 
    objective, 
):
    batch = features_all_arms.shape[0]
    k = features_all_arms.shape[1]
    device = features_all_arms.device
    n_objs = beta.shape[-1]
    #calculate mean, n=batch, k=n_arms, f=feature_dim
    mean = torch.einsum('nkf,fd->nkd', features_all_arms, beta)
    mean = mean - torch.mean(mean, dim=1, keepdim=True)

    #generate zs
    z = torch.normal(0,1,size=(batch, k, num_zs, n_objs), device = device) 
    var = torch.einsum('nkc,cfd->nkfd', features_all_arms, cov)
    
    #calculate variance
    var = torch.sqrt(torch.einsum('nkfd,nkf->nkd', var, features_all_arms))
    sigma_z = torch.einsum('nkd,nkzd->nkzd', var, z)

    #calculate mu + sigma*z
    mu = torch.einsum('nkd,nkzd->nkzd', mean, torch.ones(batch, k, num_zs, n_objs, device=device))
    value =  mu + sigma_z
    maxes = objective(value)
    maxes = torch.max(value, dim=1).values
    return torch.mean(maxes)


def get_cov(MDP, sigma, probs, features_list, cur_step, boost=1, obj=0, treat=False):
    #calculate policy limit covariance
    features_all_arms_list = []
    
    for i, features in enumerate(features_list):
        features_all_arms = MDP.features_all_arms(features, 0)
        features_all_arms_list.append(features_all_arms / (MDP.s2[cur_step + i][obj]) ** 0.5)
    
    features_all_arms = torch.cat(features_all_arms_list, dim=0).to(probs.device)
    weighted_contexts = torch.unsqueeze(probs, 2) * features_all_arms

    #n = batch, k = n_arms, f,c = feature_dim
    #f and c are both feature dimension, but einsum doesn't allow same letters
    cov = torch.einsum('nkf,nck->fc', weighted_contexts, torch.transpose(features_all_arms, 2,1)) 

    # calculate smoothed covariance
    if MDP.use_precision == False:
        smoothed_inv = torch.linalg.inv(boost*cov + torch.linalg.inv(sigma))
        return smoothed_inv @ ((boost * cov) @ sigma ) 
    elif MDP.use_precision == True:
        smoothed_inv = torch.linalg.inv(boost*cov + sigma)
        return smoothed_inv @ ((boost * cov) @ torch.linalg.inv(sigma))
    
