from aexgym.agent.linear.base_agent import LinearAgent
import torch 
from torch.nn import functional as F


class LinearUCB(LinearAgent):

    def __init__(self, model, name, alpha = 1.0, **kwargs):
        super().__init__(model, name)
        self.alpha = alpha
    
    def forward(self, beta, sigma, contexts=None, action_contexts = None, objective = None):
        
        #sample betas from prior
        true_batch_size = contexts.shape[0]
        batch_size = contexts.shape[0]
        n_objs = beta.shape[-1]
        #prepare for matrix product
        features_all_arms = self.model.features_all_arms(contexts, action_contexts)
        #n = batch_size, k = n_arms, f = feature_dim
        var = torch.einsum('nkc,cfd->nkfd', features_all_arms.float(), sigma.float())
        var = self.alpha * torch.sqrt(torch.einsum('nkfd,nkf->nkd', var, features_all_arms.float()))
        #calculate variance
        fake_mc = torch.einsum('nkf,fd->nkd', features_all_arms.float(), beta.float()) + var
        indices = torch.argmax(fake_mc, axis=1).squeeze()
        actions = torch.nn.functional.one_hot(indices, num_classes=self.model.n_arms).float()
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        
        return actions


    


        






    