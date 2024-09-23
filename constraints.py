import numpy as np
import os
import torch 
import json 

from aexgym.env import ConstraintPersSyntheticEnv
from aexgym.model import PersonalizedLinearModel
from aexgym.agent import LinearTS, LinearUniform, LinearUCB, LinearRho
from aexgym.objectives import contextual_best_arm, contextual_simple_regret
from scripts.setup_script import make_uniform_prior

n_days = 5
n_arms = 10
context_len = 5
n_steps = n_days 
batch_size = 100
s2_level = 0.2
s2 = s2_level * torch.ones((n_days, 1))

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
print(device)

#personalization 

#initialize parameterss
n_objs = 1
scaling = 1 / (batch_size*10)
pers_beta, pers_sigma = make_uniform_prior(context_len*n_arms, scaling, n_objs=n_objs)
context_mu, context_var = torch.ones(context_len), 1*torch.eye(context_len)
constraint_mu, constraint_var = torch.zeros(n_arms), 1*torch.eye(n_arms)
pers_beta = torch.ones_like(pers_beta)
#initialize synthetic and agent model 
model = PersonalizedLinearModel(
    beta_0 = pers_beta, 
    sigma_0 = pers_sigma, 
    n_arms = n_arms, 
    s2 = s2,  
    n_objs=n_objs
)

#initialize synthetic environment
env = ConstraintPersSyntheticEnv(
    model = model, 
    context_mu = context_mu, 
    context_var = context_var, 
    context_len = context_len, 
    batch_size = batch_size, 
    n_steps = n_steps,
    constraint_mu = constraint_mu,
    constraint_var = constraint_var
)

#initialize agent 
agents = [
LinearUniform(model, "Linear Uniform"),
LinearTS(model, "Linear TS 0", toptwo=False, n_samples = 100, cost_weight=0.01, constraint=False),
LinearTS(model, "Linear TS 01", toptwo=False, n_samples = 100, cost_weight=0.01, constraint=True),
LinearTS(model, "Linear TS 02", toptwo=False, n_samples = 100, cost_weight=0.05, constraint=True),
LinearTS(model, "Linear TS 03", toptwo=False, n_samples = 100, cost_weight=0.1, constraint=True),
LinearTS(model, "Linear TS 04", toptwo=False, n_samples = 100, cost_weight=0.2, constraint=True),
LinearTS(model, "Linear TS 05", toptwo=False, n_samples = 100, cost_weight=0.5, constraint=True),
LinearTS(model, "Linear TS 06", toptwo=False, n_samples = 100, cost_weight=1, constraint=True),
LinearTS(model, "Linear TS 07", toptwo=False, n_samples = 100, cost_weight=5, constraint=True),
LinearTS(model, "Linear TS 08", toptwo=False, n_samples = 100, cost_weight=10, constraint=True),
LinearTS(model, "Linear TS 09", toptwo=False, n_samples = 100, cost_weight=25, constraint=True),
LinearTS(model, "Linear TS 1", toptwo=False, n_samples = 100, cost_weight=100, constraint=True),
LinearRho(model, "Linear Rho 0", lr=0.4, epochs = 20, cost_weight = 0),
LinearRho(model, "Linear Rho 001", lr=0.4, epochs = 20, cost_weight = 0.01),
LinearRho(model, "Linear Rho 005", lr=0.4, epochs = 20, cost_weight = 0.05),
LinearRho(model, "Linear Rho 01", lr=0.4, epochs = 20, cost_weight = 0.1),
LinearRho(model, "Linear Rho 04", lr=0.4, epochs = 20, cost_weight = 0.25),
LinearRho(model, "Linear Rho 05", lr=0.4, epochs = 20, cost_weight = 0.5),
LinearRho(model, "Linear Rho 06", lr=0.4, epochs = 20, cost_weight = 0.75),
LinearRho(model, "Linear Rho 07", lr=0.4, epochs = 20, cost_weight = 1),
LinearRho(model, "Linear Rho 08", lr=0.4, epochs = 20, cost_weight = 2),
LinearRho(model, "Linear Rho 09", lr=0.4, epochs = 20, cost_weight = 5),
LinearRho(model, "Linear Rho 1", lr=0.4, epochs = 20, cost_weight = 10),

]
agent_dict = {agent.name: {} for agent in agents}
agent_dict['context_len'] = context_len
agent_dict['n_days'] = n_days
agent_dict['n_arms'] = n_arms
agent_dict['batch_size'] = batch_size
agent_dict['s2'] = s2_level
agent_dict['device'] = device
agent_dict['scaling'] = scaling
for agent in agents:
    torch.manual_seed(0)
    print_probs = False
    objective = contextual_simple_regret()
    objective.weights = (0, 1)
    torch.set_printoptions(sci_mode=False)
    simple_regret_list = []
    cumul_regret_list = []
    cost_list = []
    percent_arms_correct_list = []



    for i in range(1000):
        env.reset()
        #print(env.mean_matrix)
        cumul_regret = 0
        all_contexts, cur_step = env.reset()
        beta, sigma = agent.model.reset()
        #print(beta, sigma)
        beta, sigma = beta.to(device), sigma.to(device)
        cost_regret = 0
        while env.n_steps - cur_step > 0:

            #move to device 
            state_contexts, action_contexts, eval_contexts, costs = tuple(contexts.to(device) for contexts in all_contexts)
            #train agent 
            agent.train_agent( 
                beta = beta, 
                sigma = sigma, 
                cur_step = cur_step, 
                n_steps = n_steps, 
                train_context_sampler = env.sample_train_contexts, 
                eval_contexts = eval_contexts,
                eval_action_contexts = action_contexts, 
                real_batch = batch_size, 
                print_losses=False, 
                objective=objective,
                costs=costs,
                repeats=10000
            )    
            #get probabilities
            probs = agent(
                beta = beta, 
                sigma = sigma, 
                contexts = state_contexts, 
                action_contexts = action_contexts, 
                objective = objective,
                costs = costs 
            )
        
            #print probabilities 
            if print_probs == True:
                print(agent.name, env.n_steps - cur_step, probs)
            
            #get actions and move to new state
            actions = torch.distributions.Categorical(probs).sample()
            cost_regret += (torch.mean(costs[actions]) - torch.min(costs)).item()
            #move to next environment state 
            all_contexts, sampled_rewards, sampled_features, cur_step  = env.step(
                state_contexts = state_contexts, 
                action_contexts = action_contexts, 
                actions = actions
            )

            rewards = objective(
                agent_actions = actions,
                true_rewards = env.get_true_rewards(state_contexts, action_contexts)
            )

            cumul_regret += rewards['regret']
            
            #update model state 
            beta, sigma = agent.model.update_posterior(
                beta = beta, 
                sigma = sigma, 
                rewards = sampled_rewards, 
                features = agent.model.feature_map(actions, state_contexts, action_contexts), 
                idx = cur_step-1
            )

        #get evaluation contexts and true rewards 
        eval_contexts = env.sample_eval_contexts(access=True).to(device)
        true_eval_rewards = env.get_true_rewards(eval_contexts, action_contexts)
        
        fantasy_rewards = agent.fantasize(beta, eval_contexts, action_contexts).to(device)
        agent_actions = torch.argmax(fantasy_rewards.squeeze(), dim=1)

        #calculate results from objective 
        results_dict = objective(
            agent_actions = agent_actions, 
            true_rewards = true_eval_rewards.to(device)
        )

        cumul_regret = cumul_regret / n_days
        #results_dict['regret'] = objective.weights[0] * cumul_regret + objective.weights[1] * results_dict['regret']
        
        #append results 
        percent_arms_correct_list.append(results_dict['percent_arms_correct'])
        cumul_regret_list.append(cumul_regret)
        simple_regret_list.append(results_dict['regret'])
        cost_list.append(cost_regret)

        #print results 
        if i % 50 == 0:
            print(agent.name, i)
            agent_dict[agent.name]['simple_regret'] = (np.mean(simple_regret_list), np.std(simple_regret_list))
            agent_dict[agent.name]['cumul_regret'] = (np.mean(cumul_regret_list), np.std(cumul_regret_list))
            agent_dict[agent.name]['cost_regret'] = (np.mean(cost_list), np.std(cost_list))
            agent_dict[agent.name]['percent_arms_correct'] = (np.mean(percent_arms_correct_list), np.std(percent_arms_correct_list))
            with open(f'results/cost_dict_{n_days}.json', 'w') as f:
                json.dump(agent_dict, f)


