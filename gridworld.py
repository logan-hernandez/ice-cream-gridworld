#!/usr/bin/python
import os
import numpy as np
import math
from random import random
import sys

# example policy:
# v < > v < 
# v v v v < 
# v > > v v 
# v v v v < 
# > > o < < 

action_space = ['u', 'd', 'l', 'r', 'c']

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
      
  
  def __init__(self, init_pe, init_state = [0,0], init_grid = default_grid, rewards = reward_grid, gamma = 0.9, check_intent = False):
    self.S = init_state
    self.grid = np.transpose(init_grid)
    self.pe = init_pe
    self.gamma = gamma
    # self.shop_loc = shop_loc
    self.shop_loc = self.get_shop_locations()
    self.values = {}
    self.policy_grid = np.zeros(self.grid.shape, dtype=object)
    self.value_grid = np.zeros(self.grid.shape)
    self.rewards = np.transpose(rewards)
    self.check_intent = check_intent
    self.has_earned_reward = False
    self.N_S = len(self.grid)*len(self.grid[0])
    

  def state_index(self, state):
    return len(self.grid[0])*state[0] + state[1]
    
  def next_state(self, a):
    # Nathan Chen
    #generate a next state
    next_states, probabilities = self.get_next_state_probabilities(self.S, a)
    self.S = next_states[np.random.choice(len(next_states), p=probabilities)]


  def next_state_is_valid(self, next_state):
    # e.g. next_state == [0,0]
    if next_state[0] < 0 or next_state[1] < 0 or next_state[0] >= len(self.grid[0]) or next_state[1] >= len(self.grid[1]) or self.grid[next_state[0]][next_state[1]] == 'X':
      return False
    else:
      return True  
    
  def get_harmonic_mean_of_distances(self):
    #calc the harmonic distance
    h = 0
    for shop in self.shop_loc:
      #get distance from robot to shop
      distance = ((self.S[0]-shop[0])**2 + (self.S[1]-shop[1])**2)**(0.5)
      #compute harmonic sum
      if distance == 0: # don't divide by zero!
        return 0
      h += 1/(distance)
    h = len(self.shop_loc) / h
    return h

  def get_the_observation(self):
    #Giacomo Fratus                                                                                                       
    #loc_shops = self.get_shop_locations()
    #out put the hamonic location
    h = self.get_harmonic_mean_of_distances()
    
    #add the probability
    p_floor = math.ceil(h) - h
    rand = random()
    if rand < p_floor:
      return math.floor(h)
    else:
      return math.ceil(h)


  def get_shop_locations(self):                                                                                                                 
    shops = []
    for r in range(len(self.grid)):
      row = self.grid[r]
      for c in range(len(row)):
        cell = row[c]
        if cell == 'S':
          shops.append([r,c])
    return shops
              
  def get_next_state_probabilities(self, state, a):
    #Adrik Shamlonian 
    pe = self.pe
    p_stay = pe/4
    p_up = pe/4
    p_down = pe/4
    p_left = pe/4
    p_right = pe/4 
    if a == 'u':
      p_up = 1-pe      
    elif a == 'd':
      p_down = 1-pe
    elif a == 'r':
      p_right = 1-pe    
    elif a == 'l':
      p_left = 1-pe
    else:
      p_left = 0
      p_right = 0
      p_up = 0
      p_down = 0
      p_stay = 1 
    
    next_state_left =  [state[0] - 1, state[1]    ]
    next_state_right = [state[0] + 1, state[1]    ]
    next_state_up =    [state[0]    , state[1] - 1]
    next_state_down =  [state[0]    , state[1] + 1]
    
    if not self.next_state_is_valid(next_state_left):
      p_stay += p_left
      p_left = 0
      
    if not self.next_state_is_valid(next_state_right):
      p_stay += p_right
      p_right = 0
      
    if not self.next_state_is_valid(next_state_up):
      p_stay += p_up
      p_up = 0
      
    if not self.next_state_is_valid(next_state_down):
      p_stay += p_down
      p_down = 0
    
    next_states = [next_state_left, next_state_right, next_state_up, next_state_down, state]
    probabilities = [p_left, p_right, p_up, p_down, p_stay]
    
    return next_states, probabilities
  
  
  def calc_reward(self, state, a, next_state):
    # End state reward allocation
    r = self.rewards[next_state[0]][next_state[1]]

    if self.check_intent and r > 0:
      next_state_left =  [state[0] - 1, state[1]    ]
      next_state_right = [state[0] + 1, state[1]    ]
      next_state_up =    [state[0]    , state[1] - 1]
      next_state_down =  [state[0]    , state[1] + 1]

      should_get_reward = False
      if next_state == next_state_left and a == 'l' and (not self.has_earned_reward):
        should_get_reward = True
        self.has_earned_reward = True
      if next_state == next_state_right and a == 'r' and (not self.has_earned_reward):
        should_get_reward = True
        self.has_earned_reward = True
      if next_state == next_state_up and a == 'u' and (not self.has_earned_reward):
        should_get_reward = True
        self.has_earned_reward = True
      if next_state == next_state_down and a == 'd' and (not self.has_earned_reward):
        should_get_reward = True
        self.has_earned_reward = True
      
      if not should_get_reward:
        r = 0
    
    return r
  
  def value_iteration(self, i, state):
    if i == 0:
      return [None, 0]
    
    # dynamic programming: check if already calculated for this i and state
    if (i, state[0], state[1]) not in self.values:
      best_a = None
      max_val = float('-inf')
      for a in action_space:
        expected_val = 0 # Q
        next_states, probabilities = self.get_next_state_probabilities(state, a)
        for s in range(len(next_states)):
          next_state = next_states[s]
          p = probabilities[s]
          if p > 0:
            prev_best_a, prev_max_val = self.value_iteration(i-1, next_state)
            expected_val += p * (self.calc_reward(state, a, next_state) + self.gamma*prev_max_val)
        if expected_val > max_val:
          best_a = a
          max_val = expected_val
    	# dynamic programming: save the result for this i and state
      self.values[(i, state[0], state[1])] = [best_a, max_val]
    
    return self.values[(i, state[0], state[1])]
  
  def make_optimal_policy_from_VI(self):
    max_diff = float('inf')
    i = 1
    while max_diff > 0.01:
      max_diff = 0
      for x in range(len(self.grid)):
        for y in range(len(self.grid[x])):
          best_a, max_val = self.value_iteration(i, [x, y])
          diff = abs(max_val - self.value_grid[x][y])
          if diff > max_diff:
            max_diff = diff
          self.policy_grid[x][y] = best_a
          self.value_grid[x][y] = max_val
      i += 1
    print("Used {} value iterations to find optimal policy".format(i))
  
  
  def policy_iteration(self, policy):
    # Policy Evaluation
    P_matrix = np.zeros([self.N_S, self.N_S])
    R_matrix = np.zeros([self.N_S, self.N_S])
    
    for x in range(len(self.grid)):
      for y in range(len(self.grid[0])):
        start_state_index = self.state_index([x,y])
        next_states, probabilities = self.get_next_state_probabilities([x,y], policy[x][y])
        for s in range(len(next_states)):
          next_state = next_states[s]
          p = probabilities[s]
          end_state_index = self.state_index(next_state)
          if p > 0:
            P_matrix[start_state_index][end_state_index] = p
            R_matrix[start_state_index][end_state_index] = self.calc_reward([x,y], policy[x][y], next_state)
    
    D = np.diagonal(np.dot(P_matrix, np.transpose(R_matrix)))
    V = np.dot(np.linalg.inv(np.identity(self.N_S) - self.gamma*P_matrix), D)
    
    # Policy Refinement
    new_policy = np.zeros(self.grid.shape, dtype=object)
    for x in range(len(self.grid)):
      for y in range(len(self.grid[0])):
        max_Q = float("-inf")
        best_a = None
        for a in action_space:
          Q = 0
          next_states, probabilities = self.get_next_state_probabilities([x,y], a)
          for s in range(len(next_states)):
            next_state = next_states[s]
            p = probabilities[s]
            next_state_index = self.state_index(next_state)
            if p > 0:
              Q += p * (self.calc_reward([x,y], a, next_state) + self.gamma*V[next_state_index])
          if Q > max_Q:
            max_Q = Q
            best_a = a
        new_policy[x][y] = best_a
    return new_policy
            
  def make_optimal_policy_from_PI(self):
    # init random policy
    policy = np.zeros(self.grid.shape, dtype=object)
    for x in range(len(self.grid)):
      for y in range(len(self.grid[0])):
        policy[x][y] = 'u'
        
    run = True
    i = 0
    while run:
      new_policy = self.policy_iteration(policy)
      run = False
      for x in range(len(self.grid)):
        for y in range(len(self.grid[0])):
          if not new_policy[x][y] == policy[x][y]:
            run = True
      policy = new_policy
      i += 1
            
    print("Used {} policy iterations to find optimal policy".format(i))
    self.policy_grid = policy
        
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

      print(np.transpose(grid.grid))

      print("Generating optimal policy using VI")
      grid.make_optimal_policy_from_VI()
      grid.print_directional_grid()
      grid.print_value_grid()

      print()
      print("Generating optimal policy using PI")
      grid.make_optimal_policy_from_PI()
      grid.print_directional_grid()
      grid.print_value_grid()
      # grid1 = grid_world(init_pe=0.2, gamma=0.9)

      # print(np.transpose(grid1.grid))

      # grid1.make_optimal_policy_from_VI()
      # grid1.print_directional_grid()
      # print(np.transpose(grid1.value_grid))


     # action = get_real_input()
  else:
    grid1 = grid_world(init_pe=0.2, gamma=0.9)
    print(np.transpose(grid1.grid))
  # get keyboard
    while True:
      action = input("next command (u, d, l, r, c, end) (c means stop): ")
      if (action in action_space):
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
