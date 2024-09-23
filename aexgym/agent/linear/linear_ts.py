from aexgym.agent.linear.base_agent import LinearAgent
import torch 
from torch.nn import functional as F


class LinearTS(LinearAgent):

    def __init__(self, 
                 model, 
                 name, 
                 toptwo = False, 
                 coin = 0.5, 
                 n_samples = 1000, 
                 constraint = False,
                 cost_weight = 1, 
                 **kwargs):
        super().__init__(model, name)
        self.toptwo = toptwo
        self.coin = coin
        self.n_samples = n_samples
        self.constraint = constraint
        self.cost_weight = cost_weight

    def get_batch_size(self, contexts):
        return contexts.shape[0]

    def sample_betas(self, beta, sigma, batch_size, n_samples):
        if not self.use_precision:
            mvn = torch.distributions.MultivariateNormal(beta, sigma)
        elif self.use_precision:
            mvn = torch.distributions.MultivariateNormal(beta, precision_matrix = sigma)
        return mvn.sample((batch_size, n_samples))
    
    def calc_rewards(self, betas, features, objective, costs):
        fake_mc = torch.einsum('nkf,nsfd->nksd', features.float(), betas.float())
        fake_rewards = objective(monte_carlo_rewards = fake_mc)
        if self.constraint:
            fake_rewards = torch.einsum('nks, k->nks', fake_rewards, 1 / (self.cost_weight * costs))
        return fake_rewards
    
    def forward(self, beta, sigma, contexts=None,  action_contexts = None, objective = None, costs=None):
        
        #sample betas from prior
        batch_size = self.get_batch_size(contexts)
        n_objs = beta.shape[-1]
        betas = torch.stack([self.sample_betas(beta[:, i], sigma[:, :, i], batch_size, self.n_samples) for i in range(n_objs)], dim=3)
        features = self.model.features_all_arms(contexts, action_contexts)
        
        fake_rewards = self.calc_rewards(betas, features, objective, costs)
        
        #take highest arm for each sample and context
        indices = torch.max(fake_rewards, axis=1).indices.flatten()
        
        # get empirical probability distribution for each context
        ones = torch.ones((batch_size, self.n_samples), dtype=int, device=beta.device)
        rows = (torch.arange(batch_size, dtype=int, device=beta.device).unsqueeze(1) * ones).flatten()
        counts = torch.zeros((batch_size, self.model.n_arms), dtype=int, device = beta.device)
        probs = counts.index_put_((rows, indices), torch.ones_like(indices, dtype=int, device=beta.device), accumulate=True) / self.n_samples
        if self.toptwo == True:
            probs = torch.clamp(probs, min = 0.0001, max = 0.9999) / torch.sum(probs, dim=1, keepdim=True)
            augmented_probs = probs / (1-probs)
            prob_matrix = augmented_probs.unsqueeze(1).repeat(1, self.model.n_arms, 1) 
            mask = torch.ones(self.model.n_arms, self.model.n_arms) - torch.eye(self.model.n_arms)
            
            # Expand the mask to match the dimensions of tensor_3d
            mask = mask.unsqueeze(0).to(beta.device)
            
            # Apply the mask
            prob_matrix = torch.sum(prob_matrix * mask, dim=2)
            probs = probs * (self.coin + (1-self.coin) * prob_matrix)
            probs = probs / torch.sum(probs, dim=1, keepdim=True)
        return probs
    


class DeconfoundedTS(LinearAgent):

    def __init__(self, 
                 model, 
                 name, 
                 toptwo = False, 
                 coin = 0.5, 
                 n_samples = 1000, 
                 **kwargs):
        super().__init__(model, name)
        self.toptwo = toptwo
        self.coin = coin
        self.n_samples = n_samples

    def sample_betas(self, beta, sigma, batch_size, n_samples):
        if self.use_precision == False:
            mvn = torch.distributions.MultivariateNormal(beta, sigma)
        elif self.use_precision == True:
            mvn = torch.distributions.MultivariateNormal(beta, precision_matrix = sigma)
        return mvn.sample((batch_size, n_samples))
    
    def train_agent(self, beta, sigma, cur_step, n_steps, train_context_sampler, eval_contexts, real_batch_size, print_losses=False, objective=None):
        
        
        features = torch.mean(self.model.features_all_arms(eval_contexts), dim=0).unsqueeze(0)
        batch_size = features.shape[0]
        n_objs = beta.shape[-1]
        betas = torch.stack([self.sample_betas(beta[:, i], sigma[:, :, i], batch_size, self.n_samples) for i in range(n_objs)], dim=3)
        
        fake_mc = torch.einsum('nkf,nsfd->nksd', features, betas)
        fake_rewards = objective(monte_carlo_rewards = fake_mc)
        #take highest arm for each sample and context
        indices = torch.max(fake_rewards, axis=1).indices.flatten()

        # get empirical probability distribution for each context
        ones = torch.ones((batch_size, self.n_samples), dtype=int, device=eval_contexts.device)
        rows = torch.arange(batch_size, dtype=int, device=eval_contexts.device).unsqueeze(1)
        rows = (rows*ones).flatten()
        counts = torch.zeros((batch_size, self.model.n_arms), dtype=int, device = eval_contexts.device)
        probs = counts.index_put_((rows, indices), torch.ones_like(indices, dtype=int, device=eval_contexts.device), accumulate=True) / self.n_samples
        self.probs = torch.mean(probs, dim=0)

        if self.toptwo == True:
            probs = torch.clamp(self.probs, min = 0.00001, max = 0.9999) / torch.sum(self.probs)
            augmented_probs = probs / (1-probs)
            prob_matrix = augmented_probs.unsqueeze(0).repeat(self.model.n_arms, 1)

            mask = torch.ones(self.model.n_arms, self.model.n_arms) - torch.eye(self.model.n_arms)
            # Expand the mask to match the dimensions of tensor_3d
            mask = mask.to(eval_contexts.device)

            # Apply the mask
            prob_matrix = torch.sum(prob_matrix * mask, dim=1)
            probs = probs * (self.coin + (1-self.coin) * prob_matrix)
            self.probs = probs / torch.sum(probs)


    def forward(self, beta, sigma, contexts=None, action_contexts = None, objective=None):
        batch_size = contexts.shape[0]
        return self.probs.unsqueeze(0).repeat(batch_size, 1)

    