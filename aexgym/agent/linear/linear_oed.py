from aexgym.agent.linear.base_agent import LinearAgent
import torch 
from torch.nn import functional as F

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