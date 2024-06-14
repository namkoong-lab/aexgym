import torch
from torch import nn

def make_dict(regret, percent_arms_correct):
    results_dict = {}
    results_dict['regret'] = regret
    results_dict['percent_arms_correct'] = percent_arms_correct
    return results_dict

class contextual_simple_regret(nn.Module):
    def __init__(self):
        super(contextual_simple_regret, self).__init__()

    def forward(self, fantasy_rewards, true_rewards = None):
        if true_rewards is None:
            return fantasy_rewards[:, :, :, 0]
        else:
            policy_arms = torch.argmax(fantasy_rewards[:, :, 0], dim=1)
            policy_rewards = true_rewards[:, :, 0][torch.arange(0, true_rewards.shape[0]), policy_arms]
            optimal_rewards, optimal_actions = torch.max(true_rewards[:, :, 0], axis = 1)
            regret = torch.mean(optimal_rewards - policy_rewards).item()
            percent_arms_correct = torch.sum(torch.eq(policy_arms, optimal_actions)).item() / true_rewards.shape[0]
            return make_dict(regret, percent_arms_correct)

class contextual_best_arm(nn.Module):
    def __init__(self):
        super(contextual_best_arm, self).__init__()

    def forward(self, fantasy_rewards, true_rewards=None):
        if true_rewards is None:
            return torch.mean(fantasy_rewards[:, :, :, 0], dim=0, keepdim=True)
        else:
            fake_means = torch.mean(fantasy_rewards[:, :, 0], dim=0)
            true_means = torch.mean(true_rewards[:, :, 0], dim=0)
            
            policy_arms = torch.argmax(fake_means)
            regret = (true_means.max() - true_means[policy_arms]).item()
            percent_arms_correct = 1*(true_means.argmax() == policy_arms).item()
            return make_dict(regret, percent_arms_correct)


class constraint_best_arm(nn.Module):
    def __init__(self, constraint_vals = 0):
        super(constraint_best_arm, self).__init__()
        self.constraint_vals = constraint_vals
        self.n_objs = 2
    
    def forward(self, fantasy_rewards, true_rewards=None):
        #fake rewards is batch_size x arms x samples x n_objs
        
        if true_rewards is None:
            demeaned = fantasy_rewards[:, :, :, 1] - torch.mean(fantasy_rewards[:, :, :, 1], dim=1, keepdim=True)
            
            constraints = torch.ones_like(demeaned) * self.constraint_vals
            return fantasy_rewards[:, :, :, 0] * (torch.sigmoid(10000*(demeaned - constraints))) - 1*(1-torch.sigmoid(10000*(demeaned - constraints)))
        else:
            fake_means = torch.mean(fantasy_rewards, dim=0)
            true_means = torch.mean(true_rewards, dim=0)
            
            fake_demeaned = fake_means[:, 1] - torch.mean(fake_means[:, 1], dim=0, keepdim=True)
            true_demeaned = true_means[:, 1] - torch.mean(true_means[:, 1], dim=0, keepdim=True)
            constraints = torch.ones_like(fake_demeaned) * self.constraint_vals
            policy_arms = torch.argmax(fake_means[:, 0] *1*(fake_demeaned > constraints) - 1*(1-1*(fake_demeaned > constraints)))
            true_means = true_means[:, 0]
            #print(true_means)
            regret = (true_means.max() - true_means[policy_arms]).item()
            percent_arms_correct = 1*(true_means.argmax() == policy_arms).item()
            satisfied = (true_demeaned > constraints)[policy_arms].item()
            results_dict = {}
            results_dict['regret'] = regret
            results_dict['percent_arms_correct'] = percent_arms_correct
            results_dict['satisfied'] = satisfied
            return results_dict
        

class multi_obj_best_arm(nn.Module):
    def __init__(self, n_objs, weights):
        super(multi_obj_best_arm, self).__init__()
        self.weights = weights
        self.n_objs = n_objs
    
    def forward(self, fantasy_rewards, true_rewards=None):

        if true_rewards is None:
            return torch.einsum("nksd, d -> nks", fantasy_rewards, self.weights)
        else:
            fantasy_rewards = torch.einsum("nkd, d -> nk", fantasy_rewards, self.weights)
            true_rewards = torch.einsum("nkd, d -> nk", true_rewards, self.weights)
            fake_means, true_means = torch.mean(fantasy_rewards, dim=0), torch.mean(true_rewards, dim=0)
            policy_arms = torch.argmax(fake_means)
            regret = (true_means.max() - true_means[policy_arms]).item()
            percent_arms_correct = 1*(true_means.argmax() == policy_arms).item()

            return regret, percent_arms_correct