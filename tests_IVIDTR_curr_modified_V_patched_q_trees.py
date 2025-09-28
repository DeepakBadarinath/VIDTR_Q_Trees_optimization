#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 19:00:10 2024

@author: badarinath

"""
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
import importlib
from itertools import product, combinations
import VIDTR_envs
from VIDTR_envs import GridEnv

#%%
import IVIDTR_curr_modified_V_patched_q_trees as VIDTR_module
import markov_decision_processes as mdp_module
import disjoint_box_union
import constraint_conditions
import constraint_conditions as cc
import pandas as pd                            

                                                    
#%%
                                                                                
importlib.reload(constraint_conditions)
importlib.reload(disjoint_box_union)
importlib.reload(VIDTR_module)
importlib.reload(mdp_module)
importlib.reload(VIDTR_envs)

from markov_decision_processes import MarkovDecisionProcess as MDP
from disjoint_box_union import DisjointBoxUnion as DBU
from IVIDTR_curr_modified_V_patched_q_trees import VIDTR

#%%

class VIDTR_grid:
    
    '''
    Build the algorithm environment for the VIDTR on a grid
    
    '''
    
    def __init__(self, dimensions, center, side_lengths, stepsizes, max_lengths,
                 max_complexity, goal, time_horizon, gamma, eta, rho, max_conditions = np.inf):
        
        '''
        Parameters:
        -----------------------------------------------------------------------
        dimensions : int
                     Dimension of the grid 
        
        center : np.array
                 Center of the grid
                 
        side_lengths : np.array
                       Side lengths of the grid
                       
        stepsizes : np.array
                    Stepsizes for the grid
        
        max_lengths : np.array 
                      Maximum lengths for the grid
        
        max_complexity : int
                         Maximum complexity for the tree 
        
        goal : np.array
               Location of the goal for the 2D grid problem
               
        time_horizon : int
                       Time horizon for the VIDTR problem
                       
        gamma : float
                Discount factor
        
        eta : float
              Splitting promotion constant    
        
        rho : float
              Condition promotion constant

        Stores:
        -----------------------------------------------------------------------
        envs : list[GridEnv]
               The 2D environments for the grid for the different timesteps
        
        VIDTR_MDP : markov_decision_processes
                    The Markov Decision Process represented in the algorithm
        
        algo : VIDTR_algo
               The algorithm representing VIDTR
        '''
        self.dimensions = dimensions
        self.center = center
        self.side_lengths = side_lengths
        self.stepsizes = stepsizes
        self.max_lengths = max_lengths
        self.max_complexity = max_complexity
        self.goal = goal
        self.time_horizon = time_horizon
        self.gamma = gamma
        self.eta = eta
        self.rho = rho
        
        
        self.env = GridEnv(dimensions, center, side_lengths, goal)
        self.transitions = [self.env.transition for t in range(time_horizon)]
        self.rewards = [self.env.reward for t in range(time_horizon)]
        
        self.actions = [self.env.actions for t in range(time_horizon)]          
        self.states = [self.env.state_space for t in range(time_horizon)]       
        
        self.VIDTR_MDP = MDP(dimensions, self.states, self.actions, time_horizon, gamma,
                             self.transitions, self.rewards)                    
        
        self.algo = VIDTR(self.VIDTR_MDP, max_lengths, eta, rho, max_complexity,
                          stepsizes, max_conditions = max_conditions)
        
    
    def generate_random_trajectories(self, N):
        '''
        Generate N trajectories from the VIDTR grid setup where we take a
        random action at each timestep and we choose a random initial state
        
        Returns:
        -----------------------------------------------------------------------
           obs_states : list[list]
                        N trajectories of the states observed
        
           obs_actions : list[list]
                         N trajectories of the actions observed
           
           obs_rewards : list[list]
                         N trajectories of rewards obtained                    
           
        '''
        
        obs_states = []
        obs_actions = []
        obs_rewards = []
        
        for traj_no in range(N):
            obs_states.append([])
            obs_actions.append([])
            obs_rewards.append([])
            s = np.squeeze(self.VIDTR_MDP.state_spaces[0].pick_random_point())  
            obs_states[-1].append(s)
            
            for t in range(self.time_horizon):
                
                a = random.sample(self.actions[t], 1)[0]
                r = self.rewards[t](s,a)
                
                s = self.env.move(s,a)
                obs_states[-1].append(s)
                obs_actions[-1].append(a)
                obs_rewards[-1].append(r)
                
            
        return obs_states, obs_actions, obs_rewards

#%%%%

'''
Tests GridEnv
'''

if __name__ == '__main__':
    
    dimensions = 2
    center = np.array([0, 0])
    side_lengths = np.array([6, 6])
    goal = np.array([-1, 0])
    time_horizon = 4
    gamma = 0.9
    max_lengths = [4 for t in range(time_horizon)]
    stepsizes = 0.1
    max_complexity = 2
    #eta = -100/120
    #rho = 10.2
    etas = [0.25, 0.5, 1.0, 2.0, 5.0]
    rhos = [0.01, 0.01, 0.01, 0.01, 0.01]
    dims = [dimensions for i in range(time_horizon)]
    
    max_conditions = np.inf
    
    grid_class = VIDTR_grid(dimensions, center, side_lengths,
                            stepsizes, max_lengths, max_complexity, goal,
                            time_horizon, gamma, etas, rhos, max_conditions = max_conditions)
    
    #%%                                                                            
    N = 50
    obs_states, obs_actions, obs_rewards = grid_class.generate_random_trajectories(N)
        
    #%%
    '''
    Tests VIDTR Bellman function
    '''
    # For t = 3 is the bellman map correct
    s = np.array([2,2])
    max_val = -np.inf
    best_action = grid_class.actions[0]
    
    for a in grid_class.actions[0]:
        bellman_val = grid_class.VIDTR_MDP.reward_functions[2](s,a) 
        if bellman_val > max_val:
            best_action = a
    
    best_action


#%%
    '''
    Tests for compute_optimal_policies
    '''
    points = []
    values = []                                                                  
    
    optimal_policies, optimal_values = grid_class.algo.compute_optimal_policies()
    env = GridEnv(dimensions, center, side_lengths, goal)
    
    print(grid_class.algo.MDP.state_spaces[0])
    
    for t in range(time_horizon):
        print(optimal_policies[t](np.array([0,2])))
    
    
    for t in range(time_horizon):
        env.plot_policy_2D(optimal_policies[t], title = f'Actions at time {t}')

#%%%
    '''
    Tests for storing of optimal_value_funcs
    '''
    
    print(grid_class.algo.optimal_value_funcs)
                                                                                
#%%
    '''
    Tests for printing optimal values and actions
    '''
                                                                                
    for t in range(grid_class.time_horizon):                                    
        states = disjoint_box_union.DBUIterator(grid_class.algo.MDP.state_spaces[t])
        iter_state = iter(states)                                               
    
        print('Optimal actions are')                                            
        for s in iter_state:                                                    
            print(optimal_policies[0](np.array(s)))                                 
    
        print('Optimal values are')
        states = disjoint_box_union.DBUIterator(grid_class.algo.MDP.state_spaces[t])
        iter_state = iter(states)
        for s in iter_state:
            print(f'The value at {s} is {optimal_values[0](np.array(s))}')      
    
    
    #%%
    print('Observed states is')
    print(obs_states[0])
    
    #%%                                                                        
    '''                                                                        
    Tests for compute_interpretable_policies                                         
    '''
    
    # What is the optimization problem we wish to solve here? 
    
                            
    optimal_conditions, optimal_actions = grid_class.algo.compute_interpretable_policies(obs_states=obs_states) 
                                                                               
    #%%                                                                         
                                                                                
    for t in range(time_horizon):
                                                                                
        print(f'Optimal conditions at {t} is')                                  
        print(optimal_conditions[t])                                           
        
        
        print(f'Optimal actions at {t} is')                                     
        print(optimal_actions[t])                                               
                                                                               
    #%%                                                                         
    '''                                                                        
    VIDTR - plot errors                                                        
    '''                                                                        
    grid_class.algo.plot_errors()                                              
    
    #%%
    '''
    VIDTR - get interpretable policy                                           
    '''
    for t in range(grid_class.time_horizon-1):                                   
        
        int_policy = VIDTR.get_interpretable_policy(grid_class.algo.optimal_conditions[t],
                                                    grid_class.algo.optimal_actions[t])
            
        grid_class.env.plot_policy_2D(int_policy, title=f'Int. policy at time {t}',
                                      saved_fig_name = f'modified_vidtr_plots_{t}')
        
    #%%
    
    def plot_confidence_intervals(errors_list, title, labels, figure_title):   
        
        num_methods = len(errors_list)                                         
        means = []
        half_std_devs = []

        for method_errors in errors_list:
            method_errors = np.array(method_errors)
            mean = np.mean(method_errors)
            std_dev = np.std(method_errors)                                     
            means.append(mean)                                                  
            half_std_devs.append(std_dev / 2)  # Half of the standard deviation
    
        # Create a plot with error bars (mean ± half std_dev)
        plt.figure(figsize=(8, 5))
        x = np.arange(num_methods)  # X-axis: Method indices
        plt.errorbar(x, means, yerr=half_std_devs, fmt='o', capsize=5)
        
        plt.xticks(x[:len(labels)], labels)  # Custom labels for methods
        plt.xlabel('Integration Method')
        plt.ylabel('Error')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.savefig(figure_title)
        plt.show()
    
#%%

MAXLENGTHS = [3, 4, 5]
DIMENSIONS = [2, 3, 4]

# Create an empty DataFrame
results_df = pd.DataFrame(columns=["dimension", "max_lengths", "estimator_value"])

for dimensions in DIMENSIONS:
    for max_lengths in MAXLENGTHS:
        center = np.array([0 for d in range(dimensions)])
        side_lengths = np.array([6 for d in range(dimensions)])
        goal = np.array([-1] + [0 for d in range(dimensions-1)])
        time_horizon = 3
        gamma = 0.9
        max_lengths = [max_lengths for t in range(time_horizon)]
        stepsizes = 0.1
        max_complexity = 2
        eta = -100/120
        rho = 10.2                                                              
                                                                                
        grid_class = VIDTR_grid(dimensions, center, side_lengths,
                                stepsizes, max_lengths, max_complexity, goal,
                                time_horizon, gamma, eta, rho)    
        
        # Given a function and a parameter, run different trials of the function
        
        
        N = 50
        obs_states, obs_actions, obs_rewards = grid_class.generate_random_trajectories(N)

        optimal_conditions, optimal_actions = grid_class.algo.compute_interpretable_policies(obs_states=obs_states) 
        
        val = sum(grid_class.algo.optimal_errors[t] for t in range(grid_class.algo.time_horizon)) / grid_class.algo.time_horizon

        # Append result as a new row
        results_df = results_df.append({
            "dimension": dimensions,
            "max_lengths": max_lengths,
            "estimator_value": val
        }, ignore_index=True)

results_df.to_csv("Dimensions_max_lengths_plots.csv", index=False)
        
#%%

# Given parameters
dimensions = 2
center = np.array([0, 0])
side_lengths = np.array([6, 6])
goal = np.array([-1, 0])
time_horizon = 2
gamma = 0.9
max_lengths = [5 for _ in range(time_horizon)]
stepsizes = 0.1
max_complexity = 2
trial_number = 10

# Storage for averaged errors
error_data = {}

# Given parameters
etas = [0.1, 0.5, 1.0, 2.0]
rhos = [0.1, 0.5, 1.0, 5.0]
dimensions = 2
center = np.zeros(dimensions)
side_lengths = np.full(dimensions, 6)
goal = np.array([-1, 0])
time_horizon = 2
gamma = 0.9
max_lengths = [5 for _ in range(time_horizon)]
stepsizes = 0.1
max_complexity = 2
#max_conditions = 2  # assumed from earlier context
trial_number = 10

# Create empty DataFrame
results_df = pd.DataFrame(columns=["eta", "rho", "avg_total_error"])

# Loop over (eta, rho) combinations
for eta in etas:
    for rho in rhos:
        total_errors = []
        total_bellman_errors = []

        for _ in range(trial_number):
            grid_class = VIDTR_grid(
                dimensions, center, side_lengths, stepsizes, max_lengths,
                max_complexity, goal, time_horizon, gamma, eta, rho,
                max_conditions=max_conditions
            )
            
            N = 50
            obs_states, obs_actions, obs_rewards = grid_class.generate_random_trajectories(N)
            
            grid_class.algo.compute_interpretable_policies(obs_states=obs_states)
            
            total_errors.append(grid_class.algo.total_error)

        results_df = results_df.append({
            "eta": eta,
            "rho": rho,
            "avg_total_error": np.mean(total_errors),
        }, ignore_index=True)

# Save results to CSV
results_df.to_csv("VIDTR_errors_vs_eta_rho.csv", index=False)
  
#%%
  
'''
VIDTR - tests for different max_lengths
'''

# Given parameters
maxi_lengths = [2, 3, 4, 5]
eta = -0.0001
rho = 0.0001
dimensions = 2
center = np.zeros(dimensions)
side_lengths = np.full(dimensions, 6)
goal = np.array([-1, 0])
time_horizon = 4
gamma = 0.9
stepsizes = 0.1
max_complexity = 2
max_conditions = 2  # assumed as before
trial_number = 10

# Create empty DataFrame
results_df = pd.DataFrame(columns=["max_length", "avg_total_error"])

# Loop over max_lengths
for l in maxi_lengths:
    total_errors = []
    total_bellman_errors = []

    for _ in range(trial_number):
        max_lengths = [l] * time_horizon
        grid_class = VIDTR_grid(
            dimensions, center, side_lengths, stepsizes, max_lengths,
            max_complexity, goal, time_horizon, gamma, eta, rho,
            max_conditions=max_conditions
        )
        
        N = 50
        obs_states, obs_actions, obs_rewards = grid_class.generate_random_trajectories(N)
        
        grid_class.algo.compute_interpretable_policies(obs_states=obs_states)

        total_errors.append(grid_class.algo.total_error)

    # Append average result to DataFrame
    results_df = results_df.append({
        "max_length": l,
        "avg_total_error": np.mean(total_errors)
    }, ignore_index=True)

# Save to CSV
results_df.to_csv("VIDTR_errors_vs_max_lengths.csv", index=False)

    
#%%                                                                        
'''VIDTR - tests for different sidelengths'''

eta = -3/80
rho = 1/50
dimensions = 2
center = np.array([0, 0])
goal = np.array([-1, 0])
time_horizon = 4
gamma = 0.9
stepsizes = 0.1
max_complexity = 2
max_lengths = [5 for _ in range(time_horizon)]
possible_lengths = [4, 5, 6, 7, 8]
trial_number = 10

errors_per_length = []
bellman_errors_per_length = []

for s in possible_lengths:
    total_errors = []
    total_bellman_errors = []
    
    for _ in range(trial_number):
        side_lengths = np.array([s, s])
        grid_class = VIDTR_grid(
            dimensions, center, side_lengths, stepsizes, max_lengths,
            max_complexity, goal, time_horizon, gamma, eta, rho, max_conditions = max_conditions
        )
        
        N = 50
        obs_states, obs_actions, obs_rewards = grid_class.generate_random_trajectories(N)
        
        grid_class.algo.compute_interpretable_policies(obs_states=obs_states)
        
        total_errors.append(grid_class.algo.total_error)
    
    errors_per_length.append(np.mean(total_errors))

plt.figure(figsize=(8, 5))
plt.plot(possible_lengths, errors_per_length, label='Average Total Error', marker='o', linestyle='-')
plt.xlabel('Side Lengths')
plt.ylabel('Error')
plt.title('Average Errors per Side Length')
plt.legend()

# Save the figure
plt.savefig('VIDTR_side_lengths_optimized.png')
plt.show()

#%%
'''
VIDTR - tests for different dimensions
'''

'''VIDTR - tests for different dimensions'''

eta = -3/80  
rho = 1/50
center = np.array([0, 0])
goal = np.array([-1, 0])
time_horizon = 4
gamma = 0.9
stepsizes = 0.1
max_complexity = 2
side_lengths = np.array([6, 6])
max_lengths = [5 for _ in range(time_horizon)]
possible_dims = [3, 4, 5, 6, 7]
trial_number = 10

errors_per_dim = []
bellman_errors_per_dim = []

for dim in possible_dims:
    total_errors = []
    total_bellman_errors = []
    
    for _ in range(trial_number):
        grid_class = VIDTR_grid(
            dim, center, side_lengths, stepsizes, max_lengths,
            max_complexity, goal, time_horizon, gamma, eta, rho, max_conditions = max_conditions
        )
        N = 50
        obs_states, obs_actions, obs_rewards = grid_class.generate_random_trajectories(N)
        grid_class.algo.compute_interpretable_policies(obs_states=obs_states)
        
        total_errors.append(grid_class.algo.total_error)

    errors_per_dim.append(np.mean(total_errors))

x = np.arange(len(possible_dims))
width = 0.4

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, errors_per_dim, width, label='Total Error', color='b', alpha=0.6)
plt.bar(x + width/2, bellman_errors_per_dim, width, label='Total Bellman Error', color='r', alpha=0.6)
plt.xticks(x, possible_dims)
plt.xlabel('Dimensions')
plt.ylabel('Error')
plt.title('Average Errors per Dimension')
plt.legend()

# Save the figure
plt.savefig('VIDTR_dimensions_optimized_plot.png')
plt.show()