import torch 
from torch import Tensor
from aexgym.env.base_env import BaseSyntheticEnv

class PersContextSampler:
    def __init__(self, context_mu, context_var, context_len, batch_size):
        
        self.context_len = context_len 
        self.batch_size = batch_size
        #context prior
        self.context_mu = context_mu
        self.context_var = context_var

    def sample_state_contexts(self, i=0):
        mvn = torch.distributions.MultivariateNormal(self.context_mu, self.context_var)
        self.contexts = mvn.sample((self.batch_size,))
        return self.contexts
    
    def sample_train_contexts(self, i=0, **kwargs):
        return self.contexts

    def sample_eval_contexts(self, access = False):
        if access:
            mvn = torch.distributions.MultivariateNormal(self.context_mu, self.context_var)
            eval_contexts = mvn.sample((self.batch_size,))
        else: 
            eval_contexts = self.contexts 
        return eval_contexts
    
    def sample_action_contexts(self):
        return torch.tensor([0])


class PersSyntheticEnv(PersContextSampler, BaseSyntheticEnv):
    def __init__(self, model, context_mu, context_var, context_len, batch_size, n_steps):
        PersContextSampler.__init__(self, context_mu, context_var, context_len, batch_size)
        BaseSyntheticEnv.__init__(self, model, n_steps)

        
    


        