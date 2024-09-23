"""Base environment class for simulating contextual bandit problems"""

import abc 
import torch
from typing import Optional, Union 
from torch import Tensor
from aexgym.model import BaseModel, BaseLinearModel

class BaseContextualEnv:
    """
    Superclass that is used to structure a contextual bandit environment. 

    This class is structured around a Gym environment and the primary functions of this class 
    are as follows: 

    - sample_*_contexts(): samples a batch of contexts which could be period dependent
    - sample_arms(): provides a reward given an arm choice on a context 
    - get_all_rewards(): provides rewards for all arm choices given a context
    - step(): takes in context/action pairs, gets rewards, and outputs new contexts as next state  
    
    """

    def __init__(self, n_steps: int):
        self.n_steps = n_steps
        
    @abc.abstractmethod
    def sample_state_contexts(self, i: Optional[int] = 0):
        """ Samples contexts for given period
        
        Args:
            i (int): period (if relevant)
        
        Returns:
            contexts (tensor): 2D (batch x context_len) tensor of contexts
        """
        
        pass
    
    @abc.abstractmethod
    def sample_train_contexts(self, 
                              i: Optional[int] = 0, 
                              repeat: Optional[int] = 10):
        """ Samples contexts to feed into policy for training
        
        Args:
            i (int): period (if relevant)
            repeat (int): number of times to repeat each context (to stabilize training)

        Returns:
            contexts (tensor): 2D (batch x context_len) tensor of contexts
        """
        pass
    
    @abc.abstractmethod
    def sample_eval_contexts(self, 
                             access: Optional[bool] =False):
        """ Samples contexts from true distribution to evaluate policy
        
        Args:
            access (bool): whether policy has access to true context distribution
                           (used during training to approximate expectation)

        Returns:
            contexts (tensor): 2D (batch x context_len) tensor of contexts
        """
        pass

    @abc.abstractmethod
    def sample_action_contexts(self):
        """ Returns features associated with each arm 

        Returns:
            contexts (tensor): 2D (n_arms x action_context_len) tensor of contexts
        """
        pass

    @abc.abstractmethod
    def sample_costs(self):
        """ Returns costs associated with each arm or context/arm pair

        Returns:
            costs (tensor): 1D (n_arms) tensor costs
        """
        pass
    

    @abc.abstractmethod
    def sample_arms(self, 
                    actions: Tensor, 
                    contexts: Tensor,
                    action_contexts: Tensor):
            
        """ Samples rewards for given arms and contexts

        Args:
            actions (tensor): 1D tensor of arm indices
            contexts (tensor): 2D (batch x context_len) tensor of contexts
            action_contexts (tensor): 2D (n_arms x action_context_len) tensor of action contexts

        Returns:
            rewards (tensor): 1D tensor of rewards
            features (tensor): 2D (batch x feature_len) tensor of feature vectors
        """
        
        pass

    def get_true_rewards(self, 
                        contexts: Tensor, 
                        action_contexts: Tensor):
        """ Gets rewards for all eligible arms given a context. 
        This is useful to calculate regret or other metrics that depend on relationships 
        between different arms. 

        Args:
            contexts (tensor): 2D (batch x context_len) tensor of contexts 

        Returns: 
            all_rewards (tensor): 2D (batch x n_arms) tensor of rewards
        
        """
        pass 

    def step(self, 
             state_contexts: Tensor, 
             action_contexts: Tensor,
             actions: Tensor):
        """
        Implements step function that defines state transitions.  
        
        Args: 
            actions (tensor): 1D tensor of indices corresponding to the arm taken for previous state contexts 
            state_contexts (tensor): 2D tensor of contexts encountered at current period
        Returns:
            all_contexts (tuple): tuple of tensors consisting of:
                state_contexts (tensor): batch of contexts to be acted upon by a policy at a given period
                train_contexts (tensor): batch of contexts used to plan ahead in RHO 
                eval_contexts (tensor): batch of contexts from context distribution over all periods, not just
                                        current period 
            state_context_rewards (tensor): (batch x n_arms) specifying the reward for each arm given a context 
            steps_remaining (int): how many periods are left in the experiment
        
        """

        

        sampled_rewards, sampled_contexts = self.sample_arms(actions, state_contexts, action_contexts, i=self.counter)

        #assert self.state_contexts is not None, "Call reset before using step method."
        self.counter+=1
        #end experiment if counter exceeds length
        
        if self.n_steps - self.counter == 0:
            return None, sampled_rewards, sampled_contexts, self.counter
        
        #get new contexts
        state_contexts = self.sample_state_contexts(i = self.counter)
        eval_contexts = self.sample_eval_contexts()
        action_contexts = self.sample_action_contexts()
        costs = self.sample_costs()

        #get new state 
        all_contexts = state_contexts, action_contexts, eval_contexts, costs
        return all_contexts, sampled_rewards, sampled_contexts, self.counter

    

    def reset(self):
        """
        Resets the environment to the initial state. 
        This function must be called before using the step function. 
        """
        self.counter = 0
        state_contexts = self.sample_state_contexts()
        eval_contexts = self.sample_eval_contexts()
        action_contexts = self.sample_action_contexts()
        costs = self.sample_costs()
        all_contexts = state_contexts, action_contexts, eval_contexts, costs
        return all_contexts, self.counter
    


class BaseSyntheticEnv(BaseContextualEnv):

    """
    Subclasses the BaseContextualEnv and provides the ability to sample rewards 
    when generating data from a linear synthetic environment. 

    Given a BaseContextualEnv object, this class samples rewards as if the 
    BaseContextualEnv specifies the underlying reward generating process. 
    """

    def __init__(self, 
                 true_env: BaseLinearModel,
                 n_steps: int):
        super().__init__(n_steps)
        self.true_env = true_env


    def sample_arms(self, 
                    actions: Tensor, 
                    contexts: Tensor,
                    action_contexts: Tensor, 
                    i: Optional[int] = 0): 
        
        batch_size = contexts.shape[0]
        features = self.true_env.feature_map(actions, contexts, action_contexts)
        self.cur_beta = self.cur_beta.to(contexts.device)
        rewards = features @ self.cur_beta + torch.normal(0, self.true_env.s2[i][0]**0.5, size=(batch_size, 1), device=contexts.device)
        return rewards, contexts
    
    def get_true_rewards(self, 
                        contexts: Tensor, 
                        action_contexts: Tensor):
        features = self.true_env.features_all_arms(contexts, action_contexts)
        true_rewards = torch.einsum('nkb,bd->nkd', features, self.cur_beta)
        return true_rewards

    def reset(self):
        if self.true_env.use_precision == False:
            self.cur_beta = torch.distributions.MultivariateNormal(self.true_env.beta_0.squeeze(), self.true_env.sigma_0.squeeze()).sample().unsqueeze(1)
        elif self.true_env.use_precision == True:
            self.cur_beta = torch.distributions.MultivariateNormal(self.true_env.beta_0.squeeze(), precision_matrix = self.true_env.sigma_0.squeeze()).sample().unsqueeze(1)
        return super().reset()