import hydra 
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
import torch
from AES.objectives import contextual_best_arm, contextual_simple_regret
from AES import run_experiment_simulation
from tqdm import tqdm
import json 
import numpy as np
from scripts.setup_script import setup_dict, make_uniform_prior
import argparse


@hydra.main(config_path = "scripts/conf", config_name = "mix_config", version_base='1.2')
def main(cfg):
    exp_id, runs, batches, device = cfg.exp_id, cfg.runs, cfg.batches, cfg.device
    print(exp_id, runs, batches, device)
    native_list = OmegaConf.to_container(cfg.env_config.metric_id_list, resolve=True)
    objective = instantiate(cfg.objective)
    
    for i in range(1, 4):
        native_list = [i+1, i]
        path = cfg.path + f'{exp_id}_{cfg.env_config.subtract}_{native_list}_{cfg.env_config.batch}.json'
        
        for policy in cfg.policy_list: 
            model_cfg = OmegaConf.select(cfg.policy_list, policy)
            env = instantiate(cfg.env_config, exp_id = exp_id, context_len = batches, n_steps = batches, metric_id_list = native_list)
            #print(env.context_len)
            env.reset()
            s2 = torch.mean(env.var_matrix, dim=1)
            print(s2.shape)
            scaling = 1 / cfg.env_config.batch
            context_len, n_arms = cfg.env_config.context_len, cfg.env_config.n_arms
            if model_cfg.type == "treat":
                prior_beta, prior_sigma = make_uniform_prior(context_len + n_arms, scaling, cfg.env_config.n_objs)
                mdp = instantiate(cfg.MDP.treatment_MDP, beta_0 = prior_beta, sigma_0 = prior_sigma, s2=s2, n_objs = cfg.env_config.n_objs)
            elif model_cfg.type == "mixed":
                prior_beta, prior_sigma = make_uniform_prior(context_len + n_arms + context_len * n_arms, scaling, cfg.env_config.n_objs)
                mdp = instantiate(cfg.MDP.mixed_MDP, beta_0 = prior_beta, sigma_0 = prior_sigma, s2=s2, n_objs = cfg.env_config.n_objs)
            
            policy = instantiate(model_cfg, MDP = mdp)
            regret_list = []
            best_arm_list = []
            satisfied_list = []
            torch.manual_seed(1)
            for i in tqdm(range(runs)):
                results = run_experiment_simulation(env, policy, objective, print_probs = False, standardize=True, device = device)
                regret_list.append(results['regret'])
                best_arm_list.append(results['percent_arms_correct'])
                satisfied_list.append(results['satisfied'])
                if i % 50 ==0:
                    
                    with open(path, 'r') as f:
                        dict = json.load(f)
                    results_dict = {}
                    results_dict['regret'] = np.mean(regret_list)
                    results_dict['best_arm'] = np.mean(best_arm_list)
                    results_dict['regret_std'] = np.std(regret_list)
                    results_dict['satisfied'] = np.mean(satisfied_list)
                    models_dict = dict[str(batches)]
                    models_dict[policy.name] = results_dict
                    dict[str(batches)] = models_dict
                    with open(path, 'w') as f:
                        json.dump(dict, f)


if __name__ == "__main__":
    main()