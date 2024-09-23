from aexgym.agent.linear.base_agent import LinearAgent
import torch 
from torch.nn import functional as F 
from torch.nn import Softmax
from aexgym.agent.linear.agent_utils import TabularPolicy, PolicyNet, get_cov, LinearQFn


class LinearRho(LinearAgent):

    def __init__(
        self, 
        model, 
        name = 'LinearRho', 
        epochs = 20, 
        lr=0.4, 
        scale = 1, 
        num_zs=1000,   
        weights = (0, 1),
        cost_weight = 0,
        **kwargs
    ):
        super().__init__(model, name)
        self.epochs = epochs
        self.lr = lr
        self.scale = scale
        self.num_zs = num_zs
        self.policy_net = TabularPolicy
        self.policy = None
        self.weights = weights
        self.cost_weight = cost_weight
    
    def forward(
        self, 
        beta, 
        sigma, 
        contexts=None,
        action_contexts = None, 
        objective=None,
        costs=None
    ):
        
        batch = contexts.shape[0]
        probs = self.policy(contexts)
        return probs[:batch]
    
    
    def train_agent(
        self, 
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
        costs = None,
        repeats=10000
    ):
        
        n_objs = beta.shape[-1]
        context_len = eval_contexts.shape[1]
        train_context_list = [train_context_sampler(i=i, repeat=repeats).to(beta.device) for i in range(cur_step, n_steps)]
        train_contexts = torch.cat(train_context_list, dim=0)
        train_features_all_arms = torch.cat([self.model.features_all_arms(train_contexts, eval_action_contexts) for train_contexts in train_context_list], dim=0)
        eval_features_all_arms = self.model.features_all_arms(eval_contexts, eval_action_contexts)
        horizon = n_steps - cur_step
        train_batch = int(train_contexts.shape[0] / horizon)
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
            cov = torch.stack([get_cov(self.model, sigma[:, :, i], probs, train_context_list, cur_step, boost = boost, obj=i) for i in range(n_objs)], dim=2)      
            
            #calculate simple regret term 
            simple_reg_loss = LinearQFn(beta, cov, self.num_zs, eval_features_all_arms, objective)
            
            #calcuate cumulative regret term 
            train_mean = torch.einsum('nkf,fd->nkd', train_features_all_arms, beta) 
            train_mean = train_mean - torch.mean(train_mean, dim=1, keepdim=True)
            cumul_reg_loss = horizon *  torch.mean(torch.einsum('nk, nkd->nd', probs, train_mean))
            cost_loss = torch.mean(torch.einsum('nk, k->n', probs, costs))
            loss = self.cost_weight * cost_loss -(self.weights[0] * cumul_reg_loss + self.weights[1] * simple_reg_loss)
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
    
