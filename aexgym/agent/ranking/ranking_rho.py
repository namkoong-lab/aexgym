from aexgym.agent.linear.base_agent import LinearAgent
import torch 
from torch.nn import functional as F 
from torch.nn import Softmax
from aexgym.agent.linear.agent_utils import TabularPolicy, PolicyNet, get_cov, LinearQFn
from aexgym.agent.ranking.ranking_agent import RankingAgent

class RankingRho(RankingAgent):

    def __init__(self, 
                 model, 
                 name = 'RankingRho', 
                 epochs = 20, 
                 lr=0.4, 
                 scale = 1, 
                 num_zs=1000, 
                 treat=False, 
                 msqrt=False, 
                 **kwargs):
        super().__init__(model, name)
        self.epochs = epochs
        self.lr = lr
        self.scale = scale
        self.num_zs = num_zs
        self.policy_net = TabularPolicy
        self.policy = None
        self.treat=treat
        self.msqrt = msqrt
    
    def forward(self, 
                beta, 
                sigma, 
                contexts=None,
                action_contexts = None, 
                objective=None,
                costs=None):
        
        user_contexts, item_contexts = contexts
        batch_size = user_contexts.shape[0] 
        probs = self.policy(user_contexts)[:batch_size]
        return probs
    
    
    def train_agent(self, 
                     beta, 
                     sigma, 
                     cur_step, 
                     n_steps, 
                     train_context_sampler, 
                     eval_contexts,
                     eval_action_contexts, 
                     real_batch, 
                     print_losses=False, 
                     objective=None,
                     repeats=10000):
        
        n_objs = beta.shape[-1]
        context_len = eval_contexts[0].shape[0]
        train_context_list = [train_context_sampler(i=i, repeat=repeats) for i in range(cur_step, n_steps)]
        user_context_list = [context[0] for context in train_context_list]
        user_contexts = torch.cat(user_context_list, dim=0)
        
        eval_features_all_arms = self.model.features_all_arms(eval_contexts, eval_action_contexts)
        eval_features_all_arms = eval_features_all_arms.reshape(eval_features_all_arms.shape[0]* eval_features_all_arms.shape[2], eval_features_all_arms.shape[1], eval_features_all_arms.shape[3])

        train_batch = int(user_contexts.shape[0] / (n_steps - cur_step))
        boost = real_batch / train_batch 
        
        #initialize policy and optimizer
        policy = self.policy_net(user_contexts, self.model.n_arms).to(beta.device)
        optimizer = torch.optim.Adam(policy.parameters(), lr = self.lr)

        for epoch in range(self.epochs):

            #initialize optimizer
            optimizer.zero_grad()  
            #calculate probabilities 
            probs = policy(user_contexts)
            #get fake covariance matrix
            cov = torch.stack([get_ranking_cov(self.model, sigma[:, :, i], probs, train_context_list, eval_action_contexts, cur_step, boost = boost, obj=i, treat=self.treat) for i in range(n_objs)], dim=2)      
            loss = - torch.mean(LinearQFn(beta, cov, self.num_zs, eval_features_all_arms, objective, msqrt=self.msqrt)) 
            if print_losses == True:
                print(epoch, 'loss', -loss.item())
                print('policy', torch.mean(probs, dim=0))
                
            loss.backward()
            optimizer.step()
        # update policy 
        self.policy = policy




def get_ranking_cov(MDP, sigma, probs, features_list, action_contexts, cur_step, boost=1, obj=0, treat=False):
    #calculate policy limit covariance
    features_all_arms_list = []
    
    for i, features in enumerate(features_list):
        features_all_arms = MDP.features_all_arms(features, action_contexts)
        features_all_arms_list.append(features_all_arms / (MDP.s2[cur_step + i][obj]) ** 0.5)
    
    features_all_arms = torch.cat(features_all_arms_list, dim=0).to(probs.device)
    weighted_contexts = torch.einsum('nk,nksd->nksd', probs, features_all_arms)
    weighted_contexts = weighted_contexts.reshape(weighted_contexts.shape[0] * weighted_contexts.shape[2], weighted_contexts.shape[1], weighted_contexts.shape[3])
    features_all_arms = features_all_arms.reshape(features_all_arms.shape[0] * features_all_arms.shape[2], features_all_arms.shape[1], features_all_arms.shape[3])
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
