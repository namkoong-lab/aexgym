import torch 
import json

def make_uniform_prior(len, scaling, n_objs):
    betas = torch.stack([torch.zeros(len) for _ in range(n_objs)], dim=1)
    sigmas = torch.stack([scaling*torch.eye(len) for _ in range(n_objs)], dim=2)
    return betas, sigmas

def setup_dict(path, n_days, batch, exp_id, metric_id_list, seed, **kwargs):
        dict = {day:{} for day in range(n_days, 1, -1)}
        dict['exp_id'] = exp_id
        dict['metric_id_list'] = metric_id_list
        dict['batch_size'] = batch
        dict['seed'] = seed
        print(dict)
        with open(path, 'w') as f:
            json.dump(dict, f)


