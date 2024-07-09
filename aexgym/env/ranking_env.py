import torch
from typing import Optional, Union 
from torch import Tensor
from aexgym.model import BaseModel, BaseLinearModel
from aexgym.env.base_env import BaseContextualEnv

class RankingContextSampler:
    def __init__(self, 
                 user_context_mu, 
                 user_context_var,
                 item_context_mu,
                 item_context_var, 
                 context_len, 
                 batch_size,
                 n_arms,
                 n_items = 5,
                 total_items = 10):
        
        self.context_len = context_len 
        self.batch_size = batch_size
        #context prior
        self.user_context_mu = user_context_mu
        self.user_context_var = user_context_var

        self.item_context_mu = item_context_mu
        self.item_context_var = item_context_var

        self.total_items = total_items
        self.n_items = n_items 
        self.n_arms = n_arms
        self.action_contexts = self.reset_action_contexts()

    def reset_action_contexts(self):
        action_context_mvn = torch.distributions.MultivariateNormal(torch.zeros(self.context_len), torch.eye(self.context_len))
        action_contexts = (self.n_items, action_context_mvn.sample((self.n_arms,)))
        return action_contexts

    def sample_contexts(self, i=0):
        user_mvn = torch.distributions.MultivariateNormal(self.user_context_mu, self.user_context_var)
        user_contexts = user_mvn.sample((self.batch_size,)) 
        item_mvn = torch.distributions.MultivariateNormal(self.item_context_mu, self.item_context_var)
        item_contexts = item_mvn.sample((self.batch_size, self.total_items))
        contexts = (user_contexts, item_contexts)
        return contexts
    
    def sample_state_contexts(self, i=0):
        self.contexts = self.sample_contexts()
        return self.contexts
    
    def sample_train_contexts(self, i=0, **kwargs):
        return self.contexts

    def sample_eval_contexts(self, access = False):
        if access:
            eval_contexts = self.sample_contexts()
        else: 
            eval_contexts = self.contexts 
        return eval_contexts
    
    def sample_action_contexts(self):
        return self.action_contexts

class RankingSyntheticEnv(RankingContextSampler, BaseContextualEnv):

    """
    Subclasses the BaseContextualEnv and provides the ability to sample rewards 
    when generating data from a linear synthetic environment. 

    Given a BaseContextualEnv object, this class samples rewards as if the 
    BaseContextualEnv specifies the underlying reward generating process. 
    """

    def __init__(self,
                 true_env: BaseLinearModel,
                 n_steps: int,
                 user_context_mu, 
                 user_context_var,
                 item_context_mu,
                 item_context_var, 
                 context_len, 
                 batch_size,
                 n_arms,
                 n_items = 5,
                 total_items = 10
                 ):
        RankingContextSampler.__init__(self, 
                                       user_context_mu, 
                                       user_context_var,
                                       item_context_mu,
                                       item_context_var,
                                       context_len,
                                       batch_size,
                                       n_arms,
                                       n_items,
                                       total_items)
        BaseContextualEnv.__init__(self, n_steps)
        self.true_env = true_env


    def sample_arms(self, 
                    actions: Tensor, 
                    contexts: Tensor,
                    action_contexts: Tensor, 
                    i: Optional[int] = 0): 
        
        features = self.true_env.feature_map(actions, contexts, action_contexts)
        features = features.reshape(-1, features.shape[-1])
        self.cur_beta = self.cur_beta.to(actions.device)
        true_rewards = features @ self.cur_beta
        rewards = true_rewards + torch.normal(0, self.true_env.s2[i][0]**0.5, size=true_rewards.shape, device=true_rewards.device)
        return rewards, features
    
    def get_true_rewards(self, 
                        contexts: Tensor, 
                        action_contexts: Tensor):
        
        features = torch.sum(self.true_env.features_all_arms(contexts, action_contexts), dim=2)
        true_rewards = torch.einsum('nkb,bd->nkd', features, self.cur_beta)
        return true_rewards

    def reset(self):
        self.action_contexts = self.reset_action_contexts()
        if self.true_env.use_precision == False:
            self.cur_beta = torch.distributions.MultivariateNormal(self.true_env.beta_0.squeeze(), self.true_env.sigma_0.squeeze()).sample().unsqueeze(1)
        elif self.true_env.use_precision == True:
            self.cur_beta = torch.distributions.MultivariateNormal(self.true_env.beta_0.squeeze(), precision_matrix = self.true_env.sigma_0.squeeze()).sample().unsqueeze(1)
        return super().reset()
    
