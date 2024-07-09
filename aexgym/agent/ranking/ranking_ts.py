from aexgym.agent.linear.linear_ts import LinearTS
from aexgym.agent.ranking.ranking_agent import RankingAgent
import torch
from torch import Tensor

class RankingTS(LinearTS, RankingAgent):
    def __init__(self, 
                 model, 
                 name, 
                 toptwo = False, 
                 coin = 0.5, 
                 n_samples = 1000, 
                 constraint = False, 
                 **kwargs):
        RankingAgent.__init__(self, model, name)
        LinearTS.__init__(self, model, name, toptwo, coin, n_samples, constraint)

    def get_batch_size(self, contexts):
        user_contexts, item_contexts = contexts
        return user_contexts.shape[0]
    
    def calc_rewards(self, betas, features, objective, costs):
        features = torch.sum(features, dim=2)
        fake_mc = torch.einsum('nkf,nsfd->nksd', features.float(), betas.float())
        fake_rewards = objective(monte_carlo_rewards = fake_mc)
        if self.constraint:
            fake_rewards = torch.einsum('nks, k->nks', fake_rewards, 1 / costs)
        return fake_rewards
    
    def fantasize(self, 
                beta: Tensor, 
                contexts: Tensor, 
                action_contexts: Tensor):
        return RankingAgent.fantasize(self, beta, contexts, action_contexts)