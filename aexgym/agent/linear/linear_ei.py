from aexgym.agent.linear.base_agent import LinearAgent
import torch 
from torch.nn import functional as F

class LinearEI(LinearAgent):

    def __init__(self, model, name, **kwargs):
        super().__init__(model, name)
    
    def forward(self, beta, sigma, contexts=None, action_contexts = None, objective = None):
        
        #sample betas from prior
        true_batch_size = contexts.shape[0]
        batch_size = contexts.shape[0]
        n_objs = beta.shape[-1]
        #prepare for matrix product
        features_all_arms = self.model.features_all_arms(contexts, action_contexts)
        #n = batch_size, k = n_arms, f = feature_dim
        var = torch.einsum('nkc,cfd->nkfd', features_all_arms.float(), sigma.float())
        var = torch.mean(self.model.s2) * torch.sqrt(torch.einsum('nkfd,nkf->nkd', var, features_all_arms.float()))
        #calculate variance
        fake_mc = torch.einsum('nkf,fd->nkd', features_all_arms.float(), beta.float())
        z = (fake_mc - torch.max(fake_mc, axis=1).values * torch.ones_like(fake_mc)) / var
        normal_dist = torch.distributions.Normal(0, 1) 
        acq_func = var * z * normal_dist.cdf(z) + var * torch.exp(normal_dist.log_prob(z)) 
        indices = torch.argmax(acq_func, axis=1).squeeze()
        actions = torch.nn.functional.one_hot(indices, num_classes=self.model.n_arms).float()
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        return actions