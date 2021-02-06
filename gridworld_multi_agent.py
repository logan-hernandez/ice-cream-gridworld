# MULTI-AGENT 

import os
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import math
from random import random
import sys
import time

# example policy:
# v < > v < 
# v v v v < 
# v > > v v 
# v v v v < 
# > > o < < 

single_agent_action_space = ['u', 'd', 'l', 'r', 'c']

class grid_world:  
  default_grid = 	[['O','O','O','O','O'],
                   ['O','X','X','O','O'],
                   ['O','O','S','O','O'],
                   ['O','X','X','O','O'],
                   ['O','O','S','O','O']]
  
  reward_grid = [[0,0,0,0,-1],
                 [0,0,0,0,-1],
                 [0,0,1,0,-1],
                 [0,0,0,0,-1],
                 [0,0,10,0,-1]]
      
  
  def __init__(self, init_pe, init_state = [[0,0], [0,1]], init_grid = default_grid, rewards = reward_grid, gamma = 0.9, check_intent = False, n_agents = 2, R_C = -1):
    assert n_agents == len(init_state)
    
    self.S = init_state
    self.grid = np.transpose(init_grid)
    self.pe = init_pe
    self.gamma = gamma
    self.rewards = np.transpose(rewards)
    self.check_intent = check_intent
    
    self.n_agents = n_agents
    self.R_C = R_C # crash penalty
    
    agent_action_spaces = [single_agent_action_space] * self.n_agents
    self.action_space = np.array(np.meshgrid(*agent_action_spaces)).T.reshape(-1,self.n_agents) # derived from https://stackoverflow.com/a/35608701
    single_agent_state_space = np.array(np.meshgrid(list(range(len(self.grid))), list(range(len(self.grid[0]))))).T.reshape(-1,2)
    agent_state_space_indices = [list(range(len(single_agent_state_space)))] * self.n_agents
    state_space_indices = np.array(np.meshgrid(*agent_state_space_indices)).T.reshape(-1,self.n_agents)
    self.state_space = [single_agent_state_space[indices] for indices in state_space_indices]
    
    self.values = np.zeros(len(self.state_space))
    self.policy = np.array([np.zeros(self.n_agents, dtype=object)] * len(self.state_space))
    
  
  def state_index(self, state):
    state_index = 0
    for agent in state:
      state_index *= len(self.grid)*len(self.grid[0])
      state_index += agent[0]*len(self.grid[0]) + agent[1]
    # assert(np.array_equal(state, self.state_space[state_index]))
    return state_index
  
  # TODO --- DONE
  def next_state(self, a):
    for state in self.S:
      state = next_state_single_agent(a)
    
  # TODO --- DONE
  def next_state_single_agent(self, a):
    # Nathan Chen
    #generate a next state
    
    next_states, probabilities = self.get_next_state_probabilities(self.S, a)
    return next_states[np.random.choice(len(next_states), p=probabilities)]

  def next_agent_state_is_valid(self, next_state):
    # e.g. next_state == [0,0]
    if next_state[0] < 0 or next_state[1] < 0 or next_state[0] >= len(self.grid[0]) or next_state[1] >= len(self.grid[1]) or self.grid[next_state[0]][next_state[1]] == 'X':
      return False
    else:
      return True

  def get_next_state_probabilities(self, state, a):
    agent_next_states = []
    agent_next_state_indices = []
    for i in range(self.n_agents):
      agent_next_states.append(self.get_next_state_probabilities_single_agent(state[i], a[i]))
      agent_next_state_indices.append(list(range(len(agent_next_states[-1][0]))))
    
    # hmm this could be more readable
    next_state_permutation_indices = np.array(np.meshgrid(*agent_next_state_indices)).T.reshape(-1,self.n_agents)
    next_states = [[agent_next_states[i][0][indices[i]] for i in range(len(indices))] for indices in next_state_permutation_indices]
    probabilities = [np.prod([agent_next_states[i][1][indices[i]] for i in range(len(indices))]) for indices in next_state_permutation_indices]
    return next_states, probabilities
  
  def get_next_state_probabilities_single_agent(self, agent_state, agent_a):
    #Adrik Shamlonian
    pe = self.pe
    p_stay = pe/4
    p_up = pe/4
    p_down = pe/4
    p_left = pe/4
    p_right = pe/4 
    if agent_a == 'u':
      p_up = 1-pe      
    elif agent_a == 'd':
      p_down = 1-pe
    elif agent_a == 'r':
      p_right = 1-pe    
    elif agent_a == 'l':
      p_left = 1-pe
    else:
      p_left = 0
      p_right = 0
      p_up = 0
      p_down = 0
      p_stay = 1 
    
    next_state_left =  [agent_state[0] - 1, agent_state[1]    ]
    next_state_right = [agent_state[0] + 1, agent_state[1]    ]
    next_state_up =    [agent_state[0]    , agent_state[1] - 1]
    next_state_down =  [agent_state[0]    , agent_state[1] + 1]
    
    if not self.next_agent_state_is_valid(next_state_left):
      p_stay += p_left
      p_left = 0
    
    if not self.next_agent_state_is_valid(next_state_right):
      p_stay += p_right
      p_right = 0
    
    if not self.next_agent_state_is_valid(next_state_up):
      p_stay += p_up
      p_up = 0
      
    if not self.next_agent_state_is_valid(next_state_down):
      p_stay += p_down
      p_down = 0
    
    next_agent_states = [next_state_left, next_state_right, next_state_up, next_state_down, agent_state]
    probabilities = [p_left, p_right, p_up, p_down, p_stay]
    
    return next_agent_states, probabilities
  
  
  def calc_reward(self, state, a, next_state):
    # End state reward allocation
    r = 0
    for agent in next_state:
      r += self.rewards[agent[0]][agent[1]]
      for other_agent in next_state:
        if agent[0] == other_agent[0] and agent[1] == other_agent[1]:
          r += self.R_C
          break # each agent counts penalty once if it is in a space with any other agent
    
    # if self.check_intent and r > 0:
      # next_state_left =  [state[0] - 1, state[1]    ]
      # next_state_right = [state[0] + 1, state[1]    ]
      # next_state_up =    [state[0]    , state[1] - 1]
      # next_state_down =  [state[0]    , state[1] + 1]
    
    return r
  
  def value_iteration(self, old_values, new_values, new_policy):    
    for state_index in range(len(self.state_space)):
      state = self.state_space[state_index]
      best_a = None
      max_val = float('-inf')
      for a in self.action_space:
        expected_val = 0 # Q
        next_states, probabilities = self.get_next_state_probabilities(state, a) # TODO
        for ns in range(len(next_states)):
          next_state = next_states[ns]
          p = probabilities[ns]
          if p > 0:
            next_state_index = self.state_index(next_state)
            expected_val += p * (self.calc_reward(state, a, next_state) + self.gamma*old_values[next_state_index])
        if expected_val > max_val:
          max_val = expected_val
          best_a = a
      new_values[state_index] = max_val
      new_policy[state_index] = best_a
      
  def make_optimal_policy_from_VI(self):
    # memory allocation
    old_values = np.zeros(len(self.state_space))
    new_values = np.zeros(len(self.state_space))
    new_policy = np.array([np.zeros(self.n_agents, dtype=object)] * len(self.state_space))
    
    max_diff = float('inf')
    i = 0
    while True:
      self.value_iteration(old_values, new_values, new_policy)
      max_diff = np.amax(np.abs(new_values - old_values))
      if max_diff <= 0.01:
        break
      # swap memory that old_values and new_values point to
      temp = old_values
      old_values = new_values
      new_values = temp
      print(i, flush=True)
      i += 1
    
    print("Used {} value iterations to find optimal policy".format(i))
    self.values = new_values
    self.policy = new_policy
  
  
  def policy_iteration(self, old_policy, new_policy, new_values):
    # Policy Evaluation
    P_matrix = np.zeros([len(self.state_space), len(self.state_space)])
    R_matrix = np.zeros([len(self.state_space), len(self.state_space)])

    for state_index in range(len(self.state_space)):
      state = self.state_space[state_index]
      a = old_policy[state_index]
      next_states, probabilities = self.get_next_state_probabilities(state, a)
      for ns in range(len(next_states)):
        next_state = next_states[ns]
        p = probabilities[ns]
        if p > 0:
          next_state_index = self.state_index(next_state)
          P_matrix[state_index][next_state_index] = p
          R_matrix[state_index][next_state_index] = self.calc_reward(state, a, next_state)
    
    D = np.diagonal(np.dot(P_matrix, np.transpose(R_matrix)))
    V = np.dot(np.linalg.inv(np.identity(len(self.state_space)) - self.gamma*P_matrix), D)
    
    # Policy Refinement
    for state_index in range(len(self.state_space)):
      state = self.state_space[state_index]
      max_Q = float("-inf")
      best_a = None
      for a in self.action_space:
        Q = 0
        next_states, probabilities = self.get_next_state_probabilities(state, a)
        for ns in range(len(next_states)):
          next_state = next_states[ns]
          p = probabilities[ns]
          if p > 0:
            next_state_index = self.state_index(next_state)
            Q += p * (self.calc_reward(state, a, next_state) + self.gamma*V[next_state_index])
        if Q > max_Q:
          max_Q = Q
          best_a = a
      new_values[state_index] = max_Q
      new_policy[state_index] = best_a
  
  def make_optimal_policy_from_PI(self):
    # memory allocation
    old_policy = np.array([np.zeros(self.n_agents, dtype=object)] * len(self.state_space))
    new_policy = np.array([np.zeros(self.n_agents, dtype=object)] * len(self.state_space))
    new_values = np.zeros(len(self.state_space))
    
    # init random policy
    for state_index in range(len(self.state_space)):
      old_policy[state_index] = ['u'] * self.n_agents
    
    i = 0
    while True:
      self.policy_iteration(old_policy, new_policy, new_values)
      if np.array_equal(new_policy, old_policy):
        break
      # swap memory that old_policy and new_policy point to
      temp = old_policy
      old_policy = new_policy
      new_policy = temp
      print(i, flush=True)
      i += 1
    
    print("Used {} policy iterations to find optimal policy".format(i))
    self.values = new_values
    self.policy = new_policy
  
  def print_grid(self):
    new_grid = []
    for c in range(len(self.grid[0])):
      new_grid.append(self.grid[:,c].copy())
    new_grid[self.S[1]][self.S[0]] = 'R'
    for row in new_grid:
      print(' '.join(row))
      
  def print_directional_grid(self):
    new_grid = []
    for c in range(len(self.policy_grid[0])):
      new_grid.append(self.policy_grid[:,c].copy())
    for row in new_grid:
      for cell in row:
        if cell == 'u':
          print('^ ', end='')
        elif cell == 'd':
          print('v ', end='')
        elif cell == 'l':
          print('< ', end='')
        elif cell == 'r':
          print('> ', end='')
        elif cell == 'c':
          print('o ', end='')
      print()
      
  def print_value_grid(self):
    new_grid = []
    for c in range(len(self.value_grid[0])):
      new_grid.append(self.value_grid[:,c].copy())
    for row in new_grid:
      for cell in row:
        print('{:.2f} '.format(cell), end='')
      print()
         
def main():
  
  reward_grid_2 = [[0,0,0,0,-1],
                 [0,0,0,0,-1],
                 [0,0,10,0,-1],
                 [0,0,0,0,-1],
                 [0,0,1,0,-1]]
  
  reward_grid_3 = [[0,0,0,0,-100],
                 [0,0,0,0,-100],
                 [0,0,1,0,-100],
                 [0,0,0,0,-100],
                 [0,0,5,0,-100]]
  
  if (len(sys.argv) > 1 and sys.argv[1] == "test"):
    grid1 = grid_world(init_pe=0.2, gamma=0.9)
    grid2 = grid_world(init_pe=0.2, rewards=reward_grid_2, gamma=0.9)
    grid3 = grid_world(init_pe=0.2, rewards=reward_grid_3, gamma=0.9)
    grids = [grid1, grid2, grid3]
    for grid in grids:
      print("Map:")
      print(np.transpose(grid.grid))
      print("Number of agents: {}".format(grid.n_agents))
      print("p_e = {}".format(grid.pe))
      print("Rewards:")
      print(np.transpose(grid.rewards))
      print("Collision Penalty R_C = {}".format(grid.R_C))
      print("gamma = {}".format(grid.gamma))

      print()
      
      print("Generating optimal policy using VI", flush=True)
      start_time = time.time()
      grid.make_optimal_policy_from_VI()
      print("Took {} seconds".format(time.time() - start_time))
      print("Policy:")
      print(grid.policy, flush=True)
      print("Values:")
      print(grid.values, flush=True)
      # grid.print_directional_grid()
      # grid.print_value_grid()

      print()
      
      print("Generating optimal policy using PI", flush=True)
      start_time = time.time()
      grid.make_optimal_policy_from_PI()
      print("Took {} seconds".format(time.time() - start_time))
      print("Policy:")
      print(grid.policy, flush=True)
      print("Values:")
      print(grid.values, flush=True)
      # grid.print_directional_grid()
      # grid.print_value_grid()
      # grid1 = grid_world(init_pe=0.2, gamma=0.9)

      # print(np.transpose(grid1.grid))

      # grid1.make_optimal_policy_from_VI()
      # grid1.print_directional_grid()
      # print(np.transpose(grid1.value_grid))
      
      print()
      print()


     # action = get_real_input()
  else:
    grid1 = grid_world(init_pe=0.2, gamma=0.9)
    print(np.transpose(grid1.grid))
  # get keyboard
    while True:
      action = input("next command (u, d, l, r, c, end) (c means stop): ")
      if (action in action_space): # TODO handle inputs for multiple agents
        os.system('cls' if os.name == 'nt' else 'clear')	# hopefully this clears terminal screen
        grid1.next_state(action)
        print("action: " + action)
        print("state: ({}, {})".format(grid1.S[1], grid1.S[0]))
        print("h: {}".format(grid1.get_harmonic_mean_of_distances()))
        print("output: {}".format(grid1.get_the_observation()))
        grid1.print_grid()
      elif (action == 'end'):
        break
      else:
        print('invalid input')

      #pass the input to the grid/robot anf get the putput
      #visulvisuals the map
  
main()
