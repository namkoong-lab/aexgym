from aexgym.agent.linear.base_agent import LinearAgent
import torch 
from torch.nn import functional as F 
from torch.nn import Softmax
from aexgym.agent.linear.agent_utils import TabularPolicy, PolicyNet, get_cov, LinearQFn


class LinearRho(LinearAgent):

    def __init__(self, 
                 model, 
                 name = 'LinearRho', 
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
        
        batch = contexts.shape[0]
        probs = self.policy(contexts)
        return torch.mean(probs, dim=0).unsqueeze(0).repeat(batch, 1)
    
    
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
        context_len = eval_contexts.shape[1]
        train_context_list = [train_context_sampler(i=i, repeat=repeats) for i in range(cur_step, n_steps)]
        train_contexts = torch.cat(train_context_list, dim=0)
        eval_features_all_arms = self.model.features_all_arms(eval_contexts, eval_action_contexts)
        
        if self.treat:
            eval_features_all_arms = eval_features_all_arms[:,:,context_len:]
            beta = beta[context_len:]
            sigma = sigma[context_len:, context_len:]

        train_batch = int(train_contexts.shape[0] / (n_steps - cur_step))
        boost = real_batch / train_batch 
        
        #initialize policy and optimizer
        policy = self.policy_net(train_contexts, self.model.n_arms).to(beta.device)
        optimizer = torch.optim.Adam(policy.parameters(), lr = self.lr)

        for epoch in range(self.epochs):

            #initialize optimizer
            optimizer.zero_grad()  
            #calculate probabilities 
            probs = policy(train_contexts)
            #get fake covariance matrix
            cov = torch.stack([get_cov(self.model, sigma[:, :, i], probs, train_context_list, cur_step, boost = boost, obj=i, treat=self.treat) for i in range(n_objs)], dim=2)      
            loss = - torch.mean(LinearQFn(beta, cov, self.num_zs, eval_features_all_arms, objective, msqrt=self.msqrt)) 
            if print_losses == True:
                print(epoch, 'loss', -loss.item())
                print('policy', torch.mean(probs, dim=0))
                
            loss.backward()
            optimizer.step()
        # update policy 
        self.policy = policy


class LinearRhoNet(LinearRho):
    def __init__(self, model, name, epochs = 60, lr=0.004, scale = 1, num_zs=64, mean = False):
        super().__init__(model, name, epochs, lr, scale, num_zs, mean)
        self.policy_net = PolicyNet

    def forward(self, beta, sigma, contexts=None, batch=1):
        return self.policy(contexts)
    
