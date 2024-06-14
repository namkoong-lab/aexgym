
import torch 
from torch import Tensor 
import numpy as np
from aexgym.env import BaseContextualEnv

class ClusterEnv(BaseContextualEnv):
    def __init__(self, cluster_dict, batch_size, cluster_batch_size, n_steps, no_duplicates = True, budget = False, reward_type = ['profit']):
        super().__init__(n_steps)
        self.batch_size = batch_size 
        self.cluster_batch_size = cluster_batch_size
        self.no_duplicates = no_duplicates
        self.cluster_dict = cluster_dict
        self.budget = budget
        for key in cluster_dict.keys():
            cluster_dict[key]['response'] = cluster_dict[key]['response'][reward_type]

    
    def sample_state_contexts(self, i=0):
        return torch.ones(self.batch_size, 1)
    
    def sample_train_contexts(self, i=0, repeat=100):
        return
    
    def sample_eval_contexts(self, access=True):
        return torch.ones(1,1)
    
    def sample_action_contexts(self):
        return self.temp_feature_list
    
    def sample_arms(self, actions: Tensor, contexts: Tensor, action_contexts: Tensor, i=0):
        rewards = []
        contexts = []
        for action in actions: 
            max_len = len(self.current_dict[int(action)]['response'])
            bootstrap_indices = np.random.choice(np.arange(max_len), size=self.cluster_batch_size, replace = True)
            #bootstrap_indices = np.arange(len(self.current_dict[int(action)]['features']), dtype=int)
            temp_rewards = torch.tensor(self.current_dict[int(action)]['response'].iloc[bootstrap_indices].values)
            temp_contexts = torch.tensor(self.current_dict[int(action)]['features'].iloc[bootstrap_indices].values)
            indices = torch.tensor(self.current_dict[int(action)]['features']['treatment'].iloc[bootstrap_indices].values)
            new_rewards, new_contexts = self.transform_features(temp_rewards, temp_contexts, indices)
            rewards.append(new_rewards)
            contexts.append(new_contexts)

        return torch.cat(rewards, dim=0), torch.cat(contexts, dim=0)
    
    def bootstrap_clusters(self, batch_size_size, dict):
        bootstrap_dict = {}
        for cluster in dict.keys():
            temp_dict = {}
            max_len = len(dict[cluster]['response'])
            valid = False
            while not valid: 
                bootstrap_indices = np.random.choice(np.arange(max_len), size=batch_size_size, replace = True)
                #bootstrap_indices = np.arange(len(dict[cluster]['features']), dtype=int)
                features = dict[cluster]['features'].iloc[bootstrap_indices]
                if len(features['treatment'].unique()) > 1:
                    valid = True 
                    temp_dict['features'] = features
                    temp_dict['response'] = dict[cluster]['response'].iloc[bootstrap_indices]
                    bootstrap_dict[cluster] = temp_dict
        return bootstrap_dict

    def get_mean_features(self, dict):
        features = []
        for key in dict.keys():
            features.append(torch.tensor(dict[key]['features'].mean().values))
        return torch.stack(features, dim=0)
    
    def transform_features(self, rewards, features, indices):
        _, list_indices = torch.sort(indices, descending=True)# Use the sorted indices to sort the other two tensors
        rewards = rewards[list_indices]
        features = features[list_indices]
        n_ones = int(torch.sum(indices).item())
        new_features = torch.cat((torch.cat((features[:n_ones], torch.zeros_like(features[:n_ones])), dim=1), torch.cat((torch.zeros_like(features[n_ones:]), features[n_ones:]), dim=1)), dim=0)
        return rewards, new_features
    
    def get_true_rewards(self):
        rewards = []
        for key in self.current_dict.keys():
            rewards.append(torch.tensor(self.current_dict[key]['response'].mean().values))
        return torch.cat(rewards, dim=0)
    
    def step(self, state_contexts: Tensor, action_contexts: Tensor, actions: Tensor):
        all_contexts, sampled_rewards, sampled_contexts, self.counter = super().step(state_contexts, action_contexts, actions)
        if self.no_duplicates and all_contexts is not None:
            state_contexts, action_contexts, eval_contexts = all_contexts
            actions = torch.sort(actions, descending=True)[0]
            for action in actions:
                self.temp_feature_list = torch.cat((self.temp_feature_list[: action], self.temp_feature_list[action + 1:]), dim=0)
            all_contexts = state_contexts, self.sample_action_contexts(), eval_contexts 
        return all_contexts, sampled_rewards, sampled_contexts, self.counter
        
    def reset(self): 
        self.current_dict = self.bootstrap_clusters(self.cluster_batch_size, self.cluster_dict)
        self.feature_list = self.get_mean_features(self.current_dict)
        self.temp_feature_list = self.feature_list
        self.true_rewards = self.get_true_rewards()
        if self.budget:
            self.costs = torch.tensor(np.random.normal(size = len(self.current_dict.keys()), loc=20, scale=10))
        return super().reset()
 

            
            