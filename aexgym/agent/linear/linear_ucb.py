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

        return torch.nn.functional.one_hot(indices, num_classes=self.model.n_arms).float()

        #take highest arm for each sample and context

        # get empirical probability distribution for each context
       
    
class LinearEI(LinearAgent):

    def __init__(self, model, name, **kwargs):
        super().__init__(model, name)
    
    def forward(self, beta, sigma, contexts=None, action_features = None, objective = None):
        
        #sample betas from prior
        true_batch_size = contexts.shape[0]
        batch_size = contexts.shape[0]
        n_objs = beta.shape[-1]
        #prepare for matrix product
        features_all_arms = self.model.features_all_arms(contexts, action_features)
        #n = batch_size, k = n_arms, f = feature_dim
        var = torch.einsum('nkc,cfd->nkfd', features_all_arms.float(), sigma.float())
        var = torch.mean(self.model.s2) * torch.sqrt(torch.einsum('nkfd,nkf->nkd', var, features_all_arms.float()))
        #calculate variance
        fake_mc = torch.einsum('nkf,fd->nkd', features_all_arms.float(), beta.float())
        z = (fake_mc - torch.max(fake_mc, axis=1).values * torch.ones_like(fake_mc)) / var
        normal_dist = torch.distributions.Normal(0, 1) 
        acq_func = var * z * normal_dist.cdf(z) + var * torch.exp(normal_dist.log_prob(z)) 
        indices = torch.argmax(acq_func, axis=1).squeeze()


        return torch.nn.functional.one_hot(indices, num_classes=action_features.shape[0]).float()
    

class LinearOED(LinearAgent):

    def __init__(self, model, name, **kwargs):
        super().__init__(model, name)
    
    def forward(self, beta, sigma, contexts=None, action_features = None, objective = None):
        
        #sample betas from prior
        true_batch_size = contexts.shape[0]
        contexts = contexts[0].unsqueeze(0)
        batch_size = contexts.shape[0]
        n_objs = beta.shape[-1]
        #prepare for matrix product
        features_all_arms = self.model.features_all_arms(contexts, action_features)
        #n = batch_size, k = n_arms, f = feature_dim
        features_all_arms = features_all_arms.squeeze()
        counter = 0
        action = 0 
        curr_min = 100000
        for feature in features_all_arms:
            _ , sigma = self.model.update_posterior(beta, sigma, torch.ones((1,1)), feature.unsqueeze(0).float(), 0)
            if torch.det(sigma.squeeze()) < curr_min:
                curr_min = torch.det(sigma.squeeze())
                action = counter
        
        return torch.nn.functional.one_hot(torch.tensor([action]), num_classes=self.model.n_arms).float()
        






    