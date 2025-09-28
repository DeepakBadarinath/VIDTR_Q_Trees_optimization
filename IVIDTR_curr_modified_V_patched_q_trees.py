#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 22:36:08 2024

@author: badarinath
"""

import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import inspect
import math

import markov_decision_processes as mdp_module
import disjoint_box_union                                                                            
import constraint_conditions as cc
                    
import inhomogeneous_interval_estimation_problem as IEP

import itertools
import sys
from typing import List, Tuple, Union

sys.setrecursionlimit(9000)  # Set a higher limit if needed
                                                                                            
#%%

mdp_module = reload(mdp_module)
disjoint_box_union = reload(disjoint_box_union)
cc = reload(cc)
IEP = reload(IEP)

from disjoint_box_union import DisjointBoxUnion as DBU    
from inhomogeneous_interval_estimation_problem import QTree, QTreeNode, UpperBoundTreeSolver, InhomogeneousUpperBoundTreeSolver

#%%

class VIDTR:
    
    def __init__(self, MDP, max_lengths,
                 etas, rhos, max_complexity,
                 stepsizes, max_conditions = math.inf):                        
        '''
        Value-based interpretable dynamic treatment regimes; Generate a tree based
        policy while solving a regularized interpretable form of the Bellmann 
        equation with complexities ranging from 1 to max_complexity.
        
        In this module we assume that the state spaces are time dependent.
        
        Parameters:
        -----------------------------------------------------------------------
        MDP : MarkovDecisionProcess
              The underlying MDP from where we want to get interpretable policies
              
        max_lengths : list[T] or int
                      The max depth of the tree upto the T timesteps
        
        etas : list[T] or int
               Volume promotion constants
               Higher this value, greater promotion in the splitting process    
                                                                               
        rhos : list[T] or int                                                            
               Complexity promotion constants                                    
               Higher this value, greater the penalization effect of the complexity 
               splitting process                                                
                                                                               
        max_complexity : int or list                                                   
                         The maximum complexity of the conditions; maximum number of 
                         and conditions present in any condition               
                                                                               
        stepsizes : list[np.array((1, MDP.states.dimension[t])) for t in range(time_horizon)] or float or int        
                    The stepsizes when we have to integrate over the DBU       
        
        max_conditions : int or list                                           
                         The maximum number of conditions per time and lengthstep
                         If None then all the conditions will be looked at     
        
            
        '''
        
        self.MDP = MDP
        self.time_horizon = self.MDP.time_horizon
        
        if type(max_lengths) == float or type(max_lengths) == int:
            max_lengths = [max_lengths for t in range(self.MDP.time_horizon)]
        
        self.max_lengths = max_lengths
        
        if type(etas) == float or type(etas) == int:
            etas = [etas for t in range(self.MDP.time_horizon)]
        
        self.etas = etas
        
        if type(rhos) == float or type(rhos) == int:
            rhos = [rhos for t in range(self.MDP.time_horizon)]

        self.rhos = rhos
        
        if type(stepsizes) == float or type(stepsizes) == int:
            stepsizes = [np.ones((1, MDP.state_spaces[t].dimension)) for t in range(self.time_horizon)]
        
        self.stepsizes = stepsizes
        
        if type(max_complexity) == int:
            max_complexity = [max_complexity for t in range(self.MDP.time_horizon)]
        
        self.max_complexity = max_complexity
        
        self.true_values = [lambda s: 0 for t in range(self.MDP.time_horizon+1)]
        
        if ((type(max_conditions) == int) or (max_conditions == math.inf)):
            max_conditions = [max_conditions for t in range(self.MDP.time_horizon)]
        
        self.max_conditions = max_conditions
        #print(f'The max conditions is {self.max_conditions}')
        
    def maximum_over_actions(self, function, t):
        
        '''
        Given a function over states and actions, find the function only over
        states.
        
        Parameters:
        -----------------------------------------------------------------------
        function : function(s,a)
                   A function over states and actions for which we wish to get 
                   the map s \to max_A f(s,a)

        Returns:
        -----------------------------------------------------------------------
        max_function : function(s)
                       s \to \max_A f(s,a) is the function we wish to get
        
        '''
        def max_function(s):
            
            max_val = -np.inf
            
            for a in self.MDP.action_spaces[t]:
                if function(np.array(s),a) > max_val:
                    max_val = function(s,a)
            
            return max_val
                    
        return max_function


    def bellman_equation(self, t):
        '''
        Return the Bellman equation for the Markov Decision Process.           
        
        Assumes we know the true values from t+1 to T.                         
        
        Parameters:                                                                
        -----------------------------------------------------------------------
        t : float                                                               
            The time at which we wish to return the Bellman function for the MDP.
                                                                               
        Returns:                                                               
        -----------------------------------------------------------------------
        bellman_function : func                                                
                           The Bellman function of the MDP for the t'th timestep.

        '''
        def bellman_map(s,a):                                                   
            
            curr_space = self.MDP.state_spaces[t]
            
            if len(self.MDP.state_spaces) <= (t+1):
                new_space = curr_space
            else:
                new_space = self.MDP.state_spaces[t+1]
                                   
            action_space = self.MDP.action_spaces[t]                           
            
            dbu_iter_class = disjoint_box_union.DBUIterator(new_space)              
            dbu_iterator = iter(dbu_iter_class)                                
            
            if (t == (self.MDP.time_horizon - 1)):
            
                return self.MDP.reward_functions[t](np.array(s), a, curr_space, action_space)
            
            else:
                
                r = self.MDP.reward_functions[t](np.array(s), a, curr_space, action_space)
                T_times_V = 0
                for s_new in dbu_iterator:
                
                    kernel_eval = self.MDP.transition_kernels[t](np.array(s_new), np.array(s), a, curr_space, action_space)
                    vals_element = self.optimal_value_funcs[t+1](np.array(s_new))
                    
                    adding_element = kernel_eval * vals_element
                    T_times_V += adding_element
                
                
                return r + self.MDP.gamma * T_times_V             
        
        return bellman_map                                                     

    
    def bellman_equation_I(self, t):
        
        '''
        Return the interpretable Bellman equation for the Markov Decision Process.
        
        Assumes we know the interpretable value function from timestep t+1 to T.
        
        Parameters:
        -----------------------------------------------------------------------
        t : float
            The time at which we wish to return the interpretable Bellman equation
        
        Returns:
        -----------------------------------------------------------------------
        int_bellman_function : func
                               The Interpretable Bellman function for the MDP for the t'th timestep
        
        '''
                
        #print(f'We ask to evaluate the bellman map at timestep {t}')
        
        def bellman_map_interpretable(s,a):                                                   
            
            curr_space = self.MDP.state_spaces[t]                                   
            
            if (len(self.MDP.state_spaces) <= (t+1)):
                new_space = curr_space
            else:
                new_space = self.MDP.state_spaces[t+1]
            
            action_space = self.MDP.action_spaces[t]                           
            
            dbu_iter_class = disjoint_box_union.DBUIterator(new_space)              
            dbu_iterator = iter(dbu_iter_class)                                
            
            if t == self.MDP.time_horizon-1:
                
                return self.MDP.reward_functions[t](np.array(s), a, curr_space, action_space)
            
            else:
                                                                                
                return self.MDP.reward_functions[t](np.array(s), a, curr_space, action_space) + self.MDP.gamma * (
                            np.sum([self.MDP.transition_kernels[t](np.array(s_new), np.array(s), a, curr_space, action_space) * self.int_value_functions[0](np.array(s_new))
                            for s_new in dbu_iterator]))                        
        
        return bellman_map_interpretable                                        

    @staticmethod                                                              
    def fix_a(f, a):
        '''
        Given a function f(s,a), get the function over S by fixing the action   
                                                                                                                                                         
        Parameters:                                                                                                                                               
        -----------------------------------------------------------------------
        f : func                                                               
            The function we wish to get the projection over                    
            
        a : type(self.MDP.actions[0])                                          
            The action that is fixed                                           
        '''
        return lambda s : f(s,a)                                               
    
    
    @staticmethod
    def redefine_function(f, s, a):                                            
        '''
        Given a function f, redefine it such that f(s) is now a                
                                                                                
        Parameters:                                                            
        -----------------------------------------------------------------------
        f : function                                                           
            Old function we wish to redefine                                   
        s : type(domain(function))                                             
            The point at which we wish to redefine f                           
        a : type(range(function))                                                
            The value taken by f at s                                          
 
        Returns:                                                                  
        -----------------------------------------------------------------------
        g : function                                                           
            Redefined function                                                 
                                                                                
        '''
        def g(state):
            if np.sum((np.array(state)-np.array(s))**2) == 0:
                return a
            else:
                return f(state)
        return g
        
    @staticmethod
    def convert_function_to_dict_s_a(f, S):
        '''
        Given a function f : S \times A \to \mathbb{R}                         
        Redefine it such that f is now represented by a dictonary              

        Parameters:                                                            
        -----------------------------------------------------------------------
        f : function                                                           
            The function that is to be redefined to give a dictonary           
            
        S : iterable version of the state space                                
            iter(DisjointBoxUnionIterator)                                     

        Returns:
        -----------------------------------------------------------------------
        f_dict : dictionary
                The function which is now redefined to be a dictonary

        '''
        f_dict = {}
        for s in S:
            f_dict[tuple(s)] = f(s)
        
        return f_dict
    
    @staticmethod
    def convert_dict_to_function(f_dict, S, default_value=0):
        '''
            
        Given a dictonary f_dict, redefine it such that we get a function f from 
        S to A

        Parameters:
        -----------------------------------------------------------------------
        f_dict : dictionary
                 The dictonary form of the function
                 
        S : iterable version of the state space
            iter(DisjointBoxUnionIterator)

        Returns:
        -----------------------------------------------------------------------
        f : func
            The function version of the dictonary

        '''
            
        def f(s):
            
            if tuple(s) in f_dict.keys():
                
                return f_dict[tuple(s)]
            
            else:
                return default_value
        
        return f
    
    def compute_optimal_policies(self):
        '''
        Compute the true value functions at the different timesteps.
        
        Stores:
        -----------------------------------------------------------------------
        optimal_values : list[function]
                         A list of length self.MDP.time_horizon which represents the 
                         true value functions at the different timesteps
        
        optimal_policies : list[function]
                           The list of optimal policies for the different timesteps 
                           for the MDP
        '''
        #zero_value = lambda s : 0
        zero_value_dicts = []
        const_action_dicts = []
        
        zero_func = lambda s : 0
        
        self.optimal_policy_funcs = [zero_func for t in range(self.MDP.time_horizon)]
        self.optimal_value_funcs = [zero_func for t in range(self.MDP.time_horizon)]
        
        
        for t in range(self.time_horizon):
            
            print(f't is {t} and the state space is')
            print(self.MDP.state_spaces[t])
            
            #Setting up this iter_class is taking a lot of time-why?
            dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
            print('DBU iter class computation done')
            state_iterator = iter(dbu_iter_class)
            
            zero_dict = {}
            const_action_dict = {}
            
            for s in state_iterator:

                print(f'State is {s}')
                zero_dict[tuple(s)] = 0
                const_action_dict[tuple(s)] = self.MDP.action_spaces[t][0]
                
            
            zero_value_dicts.append(zero_dict)
            const_action_dicts.append(const_action_dict)
        
        
        optimal_policy_dicts = const_action_dicts
        
        optimal_value_dicts = zero_value_dicts
        
                                                                         
        for t in np.arange(self.time_horizon-1, -1, -1):     
                          
            dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
            
            state_iterator = iter(dbu_iter_class)
            
            for s in state_iterator:
                
                max_val = -np.inf
                
                for a in self.MDP.action_spaces[t]:
                                                    
                    bellman_value = self.bellman_equation(t)(s,a)
                    
                    if bellman_value > max_val:                                 
                        
                        max_val = bellman_value
                        optimal_action = a
                                                                                
                        optimal_policy_dicts[t][tuple(s)] = optimal_action                       
                        optimal_value_dicts[t][tuple(s)] = max_val                        
            
            
            dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
            state_iterator = iter(dbu_iter_class)
            
            optimal_policy_func = VIDTR.convert_dict_to_function(optimal_policy_dicts[t],
                                                                 state_iterator)
            
            dbu_iter_class_1 = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
            state_iterator_new = iter(dbu_iter_class_1)
            
            optimal_value_func = VIDTR.convert_dict_to_function(optimal_value_dicts[t],
                                                                state_iterator_new)
            
            self.optimal_policy_funcs[t] = optimal_policy_func
            self.optimal_value_funcs[t] = optimal_value_func
            
        
        optimal_policy_funcs = []
        optimal_value_funcs = []
        
        for t in range(self.MDP.time_horizon):
            dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
            state_iterator = iter(dbu_iter_class)
            
            optimal_policy_funcs.append(VIDTR.convert_dict_to_function(optimal_policy_dicts[t],
                                                                       state_iterator,
                                                                       default_value=self.MDP.action_spaces[t][0]))
            
            optimal_value_funcs.append(VIDTR.convert_dict_to_function(optimal_value_dicts[t],
                                                                      state_iterator,
                                                                      default_value=0.0))
        
        self.optimal_policy_funcs = optimal_policy_funcs
        self.optimal_value_funcs = optimal_value_funcs
        
        
        return optimal_policy_funcs, optimal_value_funcs
    
    def constant_eta_function(self, t):                                         
        '''
        Return the constant \eta function for time t

        Parameters:
        -----------------------------------------------------------------------
        t : int                                                                
            Time step                                                           

        Returns:
        -----------------------------------------------------------------------
        f : function
            Constant eta function at time t                                    
                                                                                 
        '''                                                                                               
        f = lambda s,a: self.etas[t]
        return f
    
    @staticmethod
    def fixed_reward_function(t, s, a, MDP, debug = False):
        
        '''
        For the MDP as in the function parameter, return the reward function.
        
        Parameters:
        -----------------------------------------------------------------------
        t : int
            Timestep at which we return the reward function
        
        s : state_point
            The point on the state space at which we return the reward function
        
        a : action
            The action we take at this reward function
        
        MDP : MarkovDecisionProcess
              The Markov Decision Process for which we compute the fixed reward function  
        
        '''
        
        if debug:
            
            print(f'For timestep {t}, state {s}, and action {a} the reward')
            print(MDP.reward_functions[t](s, a, MDP.state_spaces[t],
                                           MDP.action_spaces[t]))
        
        return MDP.reward_functions[t](s, a, MDP.state_spaces[t],
                                       MDP.action_spaces[t])
    
    @staticmethod
    def bellman_value_function_I(t, s, a, MDP, int_policy,
                                 int_value_function_next, debug = False):
        
        '''
        For the MDP, return the interpretable bellman_value_function at timestep t.
        
        Parameters:
        -----------------------------------------------------------------------
        t : int
            The timestep at which we compute the Bellman value function.
        
        s : state_point
            The point on the state space at which we return the Bellman value function.
        
        a : action
            The action we take on the Bellman value function.
        
        MDP : MarkovDecisionProcess
              The Markov Decision Process for which we compute the Bellman value function.
        
 int_policy : func
              The interpretable policy at the next timestep.
              
        '''
        
        space = MDP.state_spaces[t]                                   
        action_space = MDP.action_spaces[t]                           
        
        if len(MDP.state_spaces) <= t+1:
            iter_space = MDP.state_spaces[t]
        else:
            iter_space = MDP.state_spaces[t+1]
                                                                                
        
        dbu_iter_class = disjoint_box_union.DBUIterator(iter_space)              
        dbu_iterator = iter(dbu_iter_class)                                
        
        if debug:
            print('Reward is')
            print(MDP.reward_functions[t](np.array(s), a, space, action_space))
            
            print('Next bits are')
            total_sum = MDP.reward_functions[t](np.array(s), a, space, action_space)
            for s_new in dbu_iterator:                                         
                
                print(f'Transition kernel for {s_new}, {s}, {a} is')
                print(MDP.transition_kernels[t](np.array(s_new), np.array(s), a, space, action_space))
                
                print('Next int_value function is')
                print(int_value_function_next(np.array(s_new)))
                
                total_sum += int_value_function_next(np.array(s_new)) * MDP.transition_kernels[t](np.array(s_new), np.array(s), a, space, action_space)
            
            print('Total sum is')
            print(total_sum)
            
        return MDP.reward_functions[t](np.array(s), a, space, action_space) + MDP.gamma * (
                np.sum([[MDP.transition_kernels[t](np.array(s_new), np.array(s), a, space, action_space) * int_value_function_next(np.array(s_new)) 
                        for s_new in dbu_iterator]]))
        
    
    @staticmethod
    def last_step_int_value_function(t, int_policy, MDP, debug = False):
        
        def last_step_val_function(s):
            
            return VIDTR.fixed_reward_function(t, s, int_policy(s),
                                               MDP, debug=debug)
        
        return last_step_val_function
    
    
    @staticmethod
    def general_int_value_function(t, int_policy, MDP,
                                   next_int_val_function, debug = False):
        
        def interpretable_value_function(s):
            return VIDTR.bellman_value_function_I(t, s, int_policy(s),
                                                  MDP, int_policy,
                                                  next_int_val_function,
                                                  debug=debug)
        
        return interpretable_value_function

    
    def derive_condition(self, sigma_vals, tau_vals, non_zero_indices, dimension):
        
        '''
        Given sigma_vals and tau_vals of dimension q, check if tau[i] >= sigma[i]. if not reverse it.
        Let non_zero_indices be a vector of dimension q such that which represents the dimensions under question.
        
        Further derive the constraint condition associated with these indices, sigma and tau values.

        Parameters:
-------------------------------------------------------------------------------
        sigma_vals : list[np.array[d_1], ..., np.array[d_q]]
                     Lower bounds for the condition
    
        tau_vals : np.array[np.array[d_1], ..., np.array[d_q]]
                   Upper bounds for the condition

              q : int
                  The number of parameters considered in the state space.
                  
        Returns:
-------------------------------------------------------------------------------
        Condition : ConstraintCondition
                    I [ sigma_0 < \vec{X}_{i0} \leq tau_0, ..., sigma_q < \vec{X}_{iq} \leq tau_{iq} ]             
        
        '''
        
        print('Sigma vals is ')
        print(sigma_vals)
        
        print('Tau vals is')
        print(tau_vals)
        
        for i in range(len(sigma_vals)):
            
            if sigma_vals[i] > tau_vals[i]:
                
                x = sigma_vals[i]
                sigma_vals[i] = tau_vals[i]
                tau_vals[i] = x
        
        print('Sigma_vals and its type is')
        print(sigma_vals, type(sigma_vals))
        
        print('Tau vals and its type is')
        print(tau_vals, type(tau_vals))
        
        bounds = np.stack((np.array(sigma_vals),
                           np.array(tau_vals)), axis=1)  # Shape: (q, 2)
        
        print('Bounds is')
        print(bounds, type(bounds))
        
        condition = cc.ConstraintConditions(dimension = dimension,
                                            non_zero_indices = non_zero_indices,
                                            bounds = bounds)
        return condition
        
    @classmethod
    def generate_tuples(self, d, q):
        domain = range(d)
        
        print(f'Dimension is {d}')
        print(f'Max freq. is {q}')
        
        all_tuples = itertools.chain.from_iterable(
            (itertools.product(domain, repeat=k) for k in range(1, q + 1))
        )
        return all_tuples
        
    def compute_interpretable_policies(self, 
                                       debug = False,
                                       obs_states=None,
                                       show_pic=False):                                                               
        
        '''                                                                    
        Compute the interpretable policies for the different length and        
        timesteps given a DBUMarkovDecisionProcess.                             
        
        Parameters:                                                            
        -----------------------------------------------------------------------
        cond_stepsizes : int or float or np.array(self.DBU.dimension)               
                         The stepsizes over the conditional DBU                     
                                                                                 
        integration_method : string
                             The method of integration we wish to use -
                             trajectory integratal versus DBU based integral
        
        integral_percent : float
                           The percent of points we wish to sample from
        
        debug : bool
                Add additional print statements to debug the plots
        
        obs_states : list[list]
                     The observed states at the different time and lengthsteps
        
        conditions_string : string
                            'all' or 'order_stats' -> Use all possible conditions by going over
                            each dimension and state_differences versus conditions given by those over
                            the order statistics
        
        Stores:
        -----------------------------------------------------------------------
        optimal_conditions :  list[list]
                              condition_space[t][l] gives the optimal condition at
                              timestep t and lengthstep l
        
        optimal_errors : list[list]
                         errors[t][l] represents the error obtained at timestep t and 
                         length step l
        
        optimal_actions : list[list]
                          optimal_intepretable_policies[t][l] denotes
                          the optimal interpretable policy obtained at 
                          time t and length l
        
        stored_DBUs :   list[list]
                        stored_DBUs[t][l] is the DBU stored at timestep t and  
                        lengthstep l for the final policy
        
        stepsizes :  np.array(self.DBU.dimension) or int or float
                     The length of the stepsizes in the different dimensions
        
        total_error : list[list]
                      total_error[t][l] := Error incurred at timestep t and lengthstep l. 

        int_policies : list[list[function]]
                       optimal_intepretable_policies[t][l] denotes
                       the optimal interpretable policy obtained at 
                       time t and length l
        
        Returns:
        -----------------------------------------------------------------------
        optimal_conditions, optimal_actions : Optimal condition spaces and optimals
                                              for the given time and lengthstep
                                              respectively
        
        '''
        
        stored_DBUs = []
        optimal_conditions = []
        optimal_actions = []
        int_policies = []
        int_value_functions = []
        
        optimal_errors = []
        
        for t in np.arange(self.MDP.time_horizon-1, -1, -1):
            
            print(f'Time is {t}')
            
            all_conditions = []
            all_condition_DBUs = []
            int_policies = [[] ,*int_policies]
            
            if self.MDP.state_spaces[t].no_of_boxes == 0:
                print('Zero state space found')
            
            all_condition_DBUs = []
            
            state_bounds = self.MDP.state_spaces[t].get_total_bounds()
            #ic(state_bounds)
                        
            total_error = 0
            total_bellman_error = 0
            
            all_condition_dicts = {}
            
            for i,c in enumerate(all_conditions):
                if c != None:
                    con_DBU = DBU.condition_to_DBU(c, self.stepsizes[t])
                    if con_DBU.no_of_boxes != 0:
                        all_condition_DBUs.append(con_DBU)
                        necc_tuple = con_DBU.dbu_to_tuple()
                        if necc_tuple not in all_condition_dicts:
                            all_condition_dicts[necc_tuple] = 1
            
            if (t != self.MDP.time_horizon -1):
            
                print(f'Optimal errors for index {t+1}')
                print(optimal_errors[0])
                
                print(f'Optimal actions for index {t+1}')
                print(optimal_actions[0])
                                                                                
            optimal_errors = [[], *optimal_errors]
            stored_DBUs = [[], *stored_DBUs]
            optimal_conditions = [[], *optimal_conditions]
            optimal_actions = [[], *optimal_actions]
            
            condition_DBUs = all_condition_DBUs
            conditions = all_conditions
            optimal_cond_DBU = None
            remaining_space = self.MDP.state_spaces[t]

            for l in range(self.max_lengths[t]-1):
                
                min_error = np.inf
                optimal_condition = None
                optimal_action = None
                no_of_null_DBUs = 0
                
                print(f'We are at timestep {t} and lengthstep {l}')
            
                # Loop over all q \leq max_complexity
                
                # Loop over actions
                
                # Loop over trajectories
                
                # Create vector U_ia(X), remember that the optim problem we have is \sum U_{ia} I[X_{it} \in R - G_{tl}], in here we pull
                # out I[x \in G_{tl}] to the U_{ia}. Compute U_{ia} accordingly, some U_{ia} can be zero if X_{it} is not in the right place
                
                # U_{iat} := max{\alpha} [r_t(X_{it}, \alpha) +\gamma P^{\alpha}_t V_{t+1}(X_{it})] - [r_t(X_{it}, A_{it}) + \gamma P^{A_{it}}_t V^I_{t+1}(X_{it})] - \eta
                
                # Finish trajectory loop
                
                # Solve optimization problem to find the optimal condition for corr. lengthstep:
                # Looks like \sum_{i=1}^N U_{ia} I[ sigma_1 < X_{i1} < tau_1, ..., sigma_q < X_{iq} < tau_q ]
                
                # Add complexity constraint
                
                # Update G_{tl} which we later use in the computations as G'_{tl} := G_{tl} \cup \hat{R}_{tl}
                
                neg_eta_function = lambda s :  -self.etas[t]
                min_error = np.inf
                optimal_condition = None
                best_non_zero_indices = []
                
                # Loop over all complexities less than equal to max_complexity
                dimension = self.MDP.state_spaces[t].dimension
                optimal_action = None
                optimal_c_tuple = None
                
                for act_no, a in enumerate(self.MDP.action_spaces[t]):
                    
                    maxim_bellman_function = lambda s: self.maximum_over_actions(self.bellman_equation(t), t)(s)
                    fixed_bellman_function = lambda s: -VIDTR.fix_a(self.bellman_equation_I(t), a=a)(s)
                    total_bellman_function = lambda s: maxim_bellman_function(s) + fixed_bellman_function(s)
                    integ_function = lambda s: total_bellman_function(s) + neg_eta_function(s)
                    U_vals = []
                    points_at_t = []
                    
                    for traj_no, traj in enumerate(obs_states):
                        
                        point = traj[t]
                        points_at_t.append(point)
                        
                        if remaining_space.is_point_in_DBU(point):    
                            U_val = integ_function(point)
                        else:
                            U_val = 0
                        
                        U_vals.append(U_val)
                    
                    
                    ubts = UpperBoundTreeSolver(U_vals, np.array(points_at_t))
                    sigma_vals, tau_vals = ubts.compute_minimizers(show_pic=show_pic, filename=f'UBTS-{t}tree_{l}length')
                    
                    complexity = 0
                    c_tuple = []
                    for i in range(len(sigma_vals)):
                        if (sigma_vals[i] < tau_vals[i]):
                            complexity = complexity + 1
                            c_tuple.append(i)
                    
                    total_cost = ubts.tree.root.thresholding_cost + self.rhos[t] * complexity
                    
                    print('Action and the index is')
                    print(a, act_no)
                    
                    print('Sigma vals is')
                    print(sigma_vals)
                    
                    print('Tau vals is')
                    print(tau_vals)
                    
                    if total_cost < min_error:
                        print(f'We set optimal action as {a} which has cost {total_cost} associated to it')
                        min_error = total_cost
                        optimal_action = a
                        optimal_c_tuple = c_tuple
                
                    
                maxim_bellman_function = lambda s: self.maximum_over_actions(self.bellman_equation(t), t)(s)
                fixed_bellman_function = lambda s: -VIDTR.fix_a(self.bellman_equation_I(t), a=optimal_action)(s)
                total_bellman_function = lambda s: maxim_bellman_function(s) + fixed_bellman_function(s)
                integ_function = lambda s: total_bellman_function(s) + neg_eta_function(s)
                U_vals = []
                points_at_t = []
                
                print(f'Optimal action is {optimal_action}')
                
                for traj_no, traj in enumerate(obs_states):
                    
                    point = traj[t]
                    points_at_t.append(point)
                    
                    if remaining_space.is_point_in_DBU(point):    
                        U_val = integ_function(point)
                    else:
                        U_val = 0
                    
                    U_vals.append(U_val)
                
                ubts = UpperBoundTreeSolver(U_vals, np.array(points_at_t))
                sigma_vals, tau_vals = ubts.compute_minimizers(show_pic=show_pic, filename=f'UBTS-{t}tree_{l}length')
                total_cost = ubts.tree.root.thresholding_cost + self.rhos[t] * len(c_tuple)
                
                print(f'Optimal Sigma vals at timestep {t} and lengthstep {l} are')
                print(sigma_vals)
                
                print('Optimal Tau vals at timestep {t} and lengthstep {l} are')
                print(tau_vals)
                
                optimal_condition = cc.ConstraintConditions(dimension, non_zero_indices = np.array(range(dimension)),
                                                           bounds = np.array([sigma_vals, tau_vals]))
                
                print('Optimal condition is')
                print(optimal_condition)
                
                optimal_cond_DBU = disjoint_box_union.DisjointBoxUnion.condition_to_DBU(optimal_condition,
                                                                                        stepsizes=self.stepsizes[t])
                
                remaining_space = remaining_space.subtract_DBUs(optimal_cond_DBU)
                total_error += min_error
                
                print(f'Timestep {t} and lengthstep {l}:')                      
                print('----------------------------------------------------------------')
                print(f'Optimal condition at timestep {t} and lengthstep {l} is {optimal_condition}')
                print(f'Optimal action at timestep {t} and lengthstep {l} is {optimal_action}')
                print(f'Optimal conditional DBU at timestep {t} and lengthstep {l} is {optimal_cond_DBU}')
                print(f'Optimal error is {min_error}')                          
                print(f'Non null DBUs = {len(condition_DBUs)} - {no_of_null_DBUs}')
                print(f'Eta is {self.etas[t]}, Rho is {self.rhos[t]}')

                
                if len(optimal_errors) == 0:
                    optimal_errors = [[min_error]]
                else:
                    optimal_errors[0].append(min_error)                         
                
                if len(stored_DBUs) == 0:
                    stored_DBUs = [[optimal_cond_DBU]]
                else:
                    stored_DBUs[0].append(optimal_cond_DBU)

                if len(optimal_conditions) == 0:
                    optimal_conditions = [[optimal_condition]]
                else:
                    optimal_conditions[0].append(optimal_condition)                
            
                if len(optimal_actions) == 0:
                    optimal_actions = [[optimal_action]]
                else:
                    optimal_actions[0].append(optimal_action)    

                # Include what happens when remaining space is NULL
        

                if (remaining_space.no_of_boxes == 0):                                    
                    print('--------------------------------------------------------------')
                    print(f'For timestep {t} we end at lengthstep {l}')
                    if l != self.max_lengths[t] - 2:
                        print('Early stopping detected')
                    
                    int_policy = VIDTR.get_interpretable_policy(optimal_conditions[0],
                                                                optimal_actions[0])
                    
                    int_policies = [int_policy] + int_policies
                    
                    
                    if t == self.MDP.time_horizon - 1:
                        
                        int_value_function = VIDTR.last_step_int_value_function(t, int_policy,
                                                                                self.MDP, debug = debug)
                    
                        int_value_functions = [int_value_function] + int_value_functions
                        self.int_value_functions = int_value_functions
                        
                        print(f'We have stored int_value_functions at timestep {t}')
                        print(self.int_value_functions)
                        
                        dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
                        print('DBU iter class computation done')
                        state_iterator = iter(dbu_iter_class)
                        
                        '''
                        Tests to check if the int_value_function is greater than the actual value function for certain state points
                        '''
                        
                        for s in state_iterator:
                            
                            print(f'Int. value function and optimal_value_function at state {s} and time {t}')
                            int_value = int_value_function(s)
                            optimal_value = self.optimal_value_funcs[t](s)
                            
                            print(int_value, optimal_value)
                            
                            if int_value > optimal_value:
                                
                                print(f'For state {s} and time {t} = T-1, we have that the int value function {int_value} > {optimal_value}')
                                int_value =  VIDTR.last_step_int_value_function(t, int_policy, self.MDP, debug=debug)(s)
                                optimal_value = self.optimal_value_funcs[t](s)
                                
                                raise ValueError(f"Int_value_function at point {s} = {int_value} is greater than optimal_value_function at time {t} and state {s}")
                                
                    else:
                        
                        interpretable_value_function = VIDTR.general_int_value_function(t, int_policy,
                                                                                        self.MDP, int_value_functions[0], debug = debug)
                        
                        int_value_functions = [interpretable_value_function] + int_value_functions
                        self.int_value_functions = int_value_functions
                        
                        dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
                        #print('DBU iter class computation done')
                        state_iterator = iter(dbu_iter_class)
                        
                        #print(f'We reach the else clause for checking {t == (self.MDP.time_horizon - 1)}')
                        '''
                        Tests to check if the int value function is greater than actual value function for some state points.
                        '''
                        for s in state_iterator:
                            
                            #print(f'Int. value function and optimal_value_function at {s} and time {t}')
                            int_value = interpretable_value_function(s)
                            optimal_value = self.optimal_value_funcs[t](s)
                            
                            #print(int_value, optimal_value)
                            
                            
                            if int_value > optimal_value:
                                
                                int_value = VIDTR.general_int_value_function(t, int_policy,
                                                                             self.MDP, int_value_functions[0],
                                                                             debug = debug)
                                
                                optimal_value = self.optimal_value_funcs[t](s)
                                
                                
                                print(f'For state {s} and time {t}, we have that the int value function {int_value} > {optimal_value}')
                                raise ValueError(f"Int_value_function at point {s} = {int_value} is greater than optimal_value_function at time {t} and state {s}")
                        
                        print(f'We have stored int_value_functions at timestep {t}')
                        print(self.int_value_functions)
                        
                    break
                              
            #Final lengthstep - We can only choose the optimal action here and we work over S - \cap_{i=1}^K S_i
            
            # What happens here is that the remaining region is now just S - G_{tl}
            # We minimize over all the possible actions that can be chosen here.
            
            min_error = np.inf
            best_action = None
            
            for act_no, a in enumerate(self.MDP.action_spaces[t]):
                
                maxim_bellman_function = lambda s: self.maximum_over_actions(self.bellman_equation(t), t)(s)
                fixed_bellman_function = lambda s: -VIDTR.fix_a(self.bellman_equation_I(t), a=a)(s)
                total_bellman_function = lambda s: maxim_bellman_function(s) + fixed_bellman_function(s)
                integ_function = lambda s: total_bellman_function(s) + neg_eta_function(s)
                total_cost = remaining_space.integrate(integ_function)
                
                if total_cost < min_error:
                    min_error = total_cost
                    optimal_action = a
            
            total_error += min_error
            
            print('--------------------------------------------------------')
            print(f'Optimal action at timestep {t} and lengthstep {l} is {optimal_action}')
            print(f'Total Optimal Error is {total_cost}')
            
            optimal_errors[0].append(min_error)
            stored_DBUs[0].append(optimal_cond_DBU)
            optimal_conditions[0].append(optimal_condition)
            optimal_actions[0].append(optimal_action)
                   
            int_policy = VIDTR.get_interpretable_policy(optimal_conditions[0],
                                                        optimal_actions[0])
            
            int_policies = [int_policy] + int_policies
            
            print('We pre-append the following')
            print(int_policy)
            
            print(f'We are at the timestep {t} and the time horizon is {self.MDP.time_horizon}')
            print(f'We check {t == (self.MDP.time_horizon - 1)}')
            
            if (t == self.MDP.time_horizon - 1):
                
                int_value_function = VIDTR.last_step_int_value_function(t, int_policy, self.MDP, debug=debug)
            
                int_value_functions = [int_value_function] + int_value_functions
                self.int_value_functions = int_value_functions
                
                print(f'We have stored int_value_functions at timestep {t}')
                print(self.int_value_functions)
                
                dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
                print('DBU iter class computation done')
                state_iterator = iter(dbu_iter_class)
                
                for s in state_iterator:
                    
                    print(f'Int. value function and optimal_value_function at {s} and time {t}')
                    int_value = int_value_function(s)
                    optimal_value = self.optimal_value_funcs[t](s)
                    
                    print(int_value, optimal_value)
                    
                    if int_value > optimal_value:
                        
                        print(f'For state {s} and time {t}, we have that the int value function {int_value} > {optimal_value}')
                        
                        int_value = VIDTR.last_step_int_value_function(t, int_policy, self.MDP, debug=debug)
                        
                        optimal_value = self.optimal_value_funcs[t](s)
                        
                        raise ValueError(f"Int_value_function at point {s} = {int_value} is greater than optimal_value_function at time {t} and state {s}")
                
            else:
                
                print(f'We reach the else clause for checking {t == (self.MDP.time_horizon - 1)}')
                
                interpretable_value_function = VIDTR.general_int_value_function(t, int_policy,
                                                                                self.MDP, int_value_functions[0], debug=debug)
                
                dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
                print('DBU iter class computation done')
                state_iterator = iter(dbu_iter_class)
                
                
                for s in state_iterator:
                    
                    print(f'Int. value function and optimal_value_function at {s} and time {t}')
                    
                    int_value = interpretable_value_function(s)
                    optimal_value = self.optimal_value_funcs[t](s)
                    
                    print(int_value, optimal_value)
                    
                    
                    if (int_value > optimal_value):
                        
                        print(f'For state {s} and time {t}, we have that the int value function {int_value} > {optimal_value}')
                        print(f' Total time is {self.MDP.time_horizon}')
                        
                        int_value = VIDTR.general_int_value_function(t, int_policy, self.MDP, int_value_functions[0], debug=debug)(s)
                        
                        optimal_value = self.optimal_value_funcs[t](s)
                        
                        
                        raise ValueError(f"Int_value_function at point {s} = {int_value} is greater than optimal_value_function at time {t} and state {s}")
                        
                int_value_functions = [interpretable_value_function] + int_value_functions
                self.int_value_functions = int_value_functions
                
                print(f'We have stored int_value_functions at timestep {t}')
                print(self.int_value_functions)
                
        self.optimal_conditions = optimal_conditions
        self.optimal_errors = optimal_errors
        self.optimal_actions = optimal_actions
        self.stored_DBUs = stored_DBUs
        self.total_bellman_error = total_bellman_error
        self.total_error = total_error
        self.int_policies = int_policies
        
        return optimal_conditions, optimal_actions
    
    @staticmethod
    def get_interpretable_policy(conditions, actions):
        '''                                                                    
        Given the conditions defining the policy, obtain the interpretable policy
        implied by the conditions.                                             
        
        Parameters:
        -----------------------------------------------------------------------
        conditions : np.array[l]
                     The conditions we want to represent in the int. policy             
        
        actions : np.array[l]
                  The actions represented in the int. policy
                                                                               
        '''

        def policy(state):
            
            for i, cond in enumerate(conditions):                               
                                                                                 
                if cond.contains_point(state):                                  
                                                                                
                    return actions[i]                                           
            
                                                                                
            return actions[0]                                                  
        
        return policy
    
    @staticmethod
    def tuplify_2D_array(two_d_array):
        two_d_list = []
        n,m = two_d_array.shape
        
        for i in range(n):
            two_d_list.append([])
            for j in range(m):
                two_d_list[-1].append(two_d_array[i,j])
        
            two_d_list[-1] = tuple(two_d_array[-1])                                 
        two_d_tuple = tuple(two_d_list)                                             
        return two_d_tuple                                                     
        
    
    def plot_errors(self):
        '''
        Plot the errors obtained after we perform the VIDTR algorithm.          

        '''
        for t in range(len(self.optimal_errors)):                                  
            plt.plot(np.arange(len(self.optimal_errors[t])), self.optimal_errors[t])
            plt.title(f'Errors at time {t}')
            plt.xlabel('Lengths')
            plt.ylabel('Errors')
            plt.show()
            
            