import torch 
from aexgym.env.base_env import BaseSyntheticEnv, BaseContextualEnv

class DOWContextSampler:
    def __init__(self, context_len, batch_size):
        self.context_len = context_len  
        self.batch_size = batch_size
    
    def sample_state_contexts(self, 
                              i=0):
        day = torch.zeros(self.context_len).unsqueeze(0)
        day[0, i] = 1
        contexts = day.repeat(self.batch_size, 1)
        return contexts
    
    def sample_train_contexts(self, 
                              i=0, 
                              repeat=100):
        day = torch.zeros(self.context_len).unsqueeze(0)
        day[0, i] = 1
        contexts = day.repeat(repeat, 1)
        return contexts
    
    def sample_eval_contexts(self, 
                             access=True):
        for j in range(0, self.context_len):
            day = torch.zeros(self.context_len).unsqueeze(0)
            day[0, j] = 1 

            if j == 0:
                eval_contexts = day
            else:
                eval_contexts = torch.cat((eval_contexts, day), dim=0)
        return eval_contexts

    def sample_action_contexts(self):
        return torch.tensor([0])

class DOWSyntheticEnv(DOWContextSampler, BaseSyntheticEnv):
    def __init__(self, dow_env, context_len, batch_size, n_steps):
        DOWContextSampler.__init__(self, context_len, batch_size)
        BaseSyntheticEnv.__init__(self, dow_env, n_steps)



        
    

    


    

        



