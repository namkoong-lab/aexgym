import torch
import pandas as pd
import json
from aexgym.env import DOWContextSampler, BaseContextualEnv
from typing import Optional, List
import numpy as np
from torch import Tensor

def make_matrices(asos_df, T, num_arms, arms, exp_id, metric_id_list=[], demean = False, subtract = False):
    mean_matrices, var_matrices = [], []
    for i, metric_id in enumerate(metric_id_list):
        exp_036afc = asos_df[(asos_df['experiment_id'] == exp_id) & (asos_df['metric_id'] == metric_id)]
        
        ts = list(exp_036afc['time_since_start'])
        arm_dat = {t:{k:{} for k in range(num_arms)} for t in range(len(ts))}

        for t, row in enumerate(exp_036afc.iterrows()):
            row = row[1]
            
            for k in range(num_arms):
                
                if k == 0:
                    arm_dat[t][k]['mean'] = row['mean_c']
                    arm_dat[t][k]['var']  = row['variance_c']
                elif k == 1:
                    arm_dat[t][k]['mean'] = row['mean_t']
                    arm_dat[t][k]['var']  = row['variance_t']
                else:
                    scale = row['mean_t'] - row['mean_c']
                    arm_dat[t][k]['mean'] = row['mean_c'] + scale * arms[i][k]
                    arm_dat[t][k]['var']  = row['variance_t']
        
        if demean:
            for t in range(T):
                mean = np.mean([arm_dat[t][k]['mean'] for k in range(num_arms)])
                for k in range(num_arms):
                    arm_dat[t][k]['mean'] -= mean 
        
        
        mean_matrix = torch.zeros((T+1, num_arms))
        var_matrix = torch.zeros((T+1, num_arms))
        for t in range(T+1):
            for k in range(num_arms):
                mean_matrix[t, k] = arm_dat[t][k]['mean']
                var_matrix[t, k] = arm_dat[t][k]['var']
        var_matrix = torch.abs(var_matrix)
        if subtract:
            for i in range(T, 0, -1):
                mean_matrix[i] = mean_matrix[i] - mean_matrix[i-1]
                var_matrix[i] = var_matrix[i] - torch.mean(var_matrix[i-1])
            var_matrix = torch.abs(var_matrix)
            mean_matrices.append(mean_matrix[1:])
            var_matrices.append(var_matrix[1:])
        elif not subtract:
            for i in range(T, 0, -1):
                mean_matrix[i] = mean_matrix[i] - torch.mean(mean_matrix[i-1])
                var_matrix[i] = var_matrix[i] - torch.mean(var_matrix[i-1])
            var_matrix = torch.abs(var_matrix)
            mean_matrices.append(mean_matrix[1:])
            var_matrices.append(var_matrix[1:])

    return torch.stack(mean_matrices, dim=2), torch.stack(var_matrices, dim=2)

def create_zs(n_arms = 10, seed=1, n_objs=3):
    np.random.seed(seed)
    zs = []
    for _ in range(n_objs):
        tmp_zs = []
        for i in range(100000):
            arms = [0, 1] + [np.random.normal(0,1) for i in range(n_arms - 2)]
            tmp_zs.append(torch.tensor(arms))
        zs.append(torch.stack(tmp_zs))
    
    return torch.stack(zs, dim=1)



class ASOS(DOWContextSampler, BaseContextualEnv):
    def __init__(self, 
                 asos_df: pd.DataFrame,
                 context_len: int,
                 batch_size: int, 
                 n_steps: int,
                 n_arms: int,
                 s2: Optional[float] = None, 
                 seed: Optional[int] = 1,
                 exp_id: Optional[int] = 1, 
                 metric_id_list: Optional[List] = [1,2,3], 
                 demean: Optional[bool] = False, 
                 subtract: Optional[bool] = False,
                 n_objs: Optional[int] = 3,
                 noise: Optional[str] = 'Gaussian'):
        DOWContextSampler.__init__(self, context_len, batch_size)
        BaseContextualEnv.__init__(self, n_steps)
        self.asos_df = asos_df
        self.subtract = subtract
        self.demean=demean
        self.exp_id = exp_id 
        self.metric_id_list = metric_id_list
        self.n_arms = n_arms
        self.n_objs = n_objs
        self.seed = seed
        self.s2 = s2
        self.zs = create_zs(n_arms=self.n_arms, seed=self.seed, n_objs=self.n_objs)
        self.z_counter = 0
        self.noise = noise
        

    def sample_arms(self, best_arms, contexts, action_contexts, i=0):
        self.mean_matrix = self.mean_matrix.to(contexts.device)
        self.var_matrix = self.var_matrix.to(contexts.device)
        
        indices = torch.argmax(contexts, dim=1)
        means = self.mean_matrix[indices, best_arms, :]
        vars = self.var_matrix[indices, best_arms, :]
        if self.noise == 'Gaussian':
            return torch.normal(means, torch.sqrt(vars))
        elif self.noise == 'Gumbel':
            euler_constant = 0.5772156649 
            zeros = torch.zeros_like(means, device=contexts.device)
            gumbel_beta = torch.sqrt(6 * vars) / torch.pi
            gumbel_mu = zeros - euler_constant * gumbel_beta
            gumbel = torch.distributions.gumbel.Gumbel(gumbel_mu, gumbel_beta)
            return means + gumbel.sample()


    def get_true_rewards(self, contexts, action_contexts):
        indices = torch.argmax(contexts, dim=1)
        return self.mean_matrix[indices]
    
    def step(self, 
             state_contexts: Tensor, 
             action_contexts: Tensor, 
             actions: Tensor, 
             sequential=False):
        """
        Implements step function that defines state transitions.  
        
        Args: 
            actions (tensor): 1D tensor of indices corresponding to the arm taken for previous state contexts 
            state_contexts (tensor): 2D tensor of contexts encountered at current period
        Returns:
            all_contexts (tuple): tuple of tensors consisting of:
                state_contexts (tensor): batch_size of contexts to be acted upon by a policy at a given period
                train_contexts (tensor): batch_size of contexts used to plan ahead in RHO 
                eval_contexts (tensor): batch_size of contexts from context distribution over all periods, not just
                                        current period 
            state_context_rewards (tensor): (batch_size x n_arms) specifying the reward for each arm given a context 
            steps_remaining (int): how many periods are left in the experiment
        
        """

        action_contexts = self.sample_action_contexts()
        if sequential:
            sampled_rewards = self.sample_arms(actions, state_contexts, action_contexts, i=int(self.counter  / self.batch_size))
        else:
            sampled_rewards = self.sample_arms(actions, state_contexts, action_contexts, i=self.counter)
        all_arm_rewards = self.get_true_rewards(state_contexts, action_contexts)

        #assert self.state_contexts is not None, "Call reset before using step method."
        self.counter+=1
        #end experiment if counter exceeds length
        if sequential:
            if self.n_steps - int(self.counter / self.batch_size) == 0:
                return None, sampled_rewards, all_arm_rewards, self.counter
        elif not sequential:
            if self.n_steps - self.counter == 0:
                return None, sampled_rewards, all_arm_rewards, self.counter
        #get new contexts
        if sequential:
            state_contexts = self.sample_state_contexts(i = int(self.counter / self.batch_size))
        elif not sequential:   
            state_contexts = self.sample_state_contexts(i = self.counter)
        eval_contexts = self.sample_eval_contexts()

        #get new state 
        all_contexts = state_contexts, action_contexts, eval_contexts
        return all_contexts, sampled_rewards, all_arm_rewards, self.counter
    
    def reset(self):
        self.mean_matrix, self.var_matrix = make_matrices(self.asos_df, self.context_len, self.n_arms, self.zs[self.z_counter], exp_id = self.exp_id, metric_id_list = self.metric_id_list, demean=self.demean, subtract = self.subtract)
        #print(self.mean_matrix)
        if self.s2 is not None:
            self.var_matrix = self.s2*torch.ones_like(self.var_matrix)
        self.true_mean = torch.mean(self.mean_matrix, dim=0)
        self.z_counter+=1
        return super().reset()