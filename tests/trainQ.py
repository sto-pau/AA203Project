import numpy as np
from networkQapprox import *
import csv
import pandas as pd

def readCSV(path):
    #from this data we obtain the dataset and the variables
    df = pd.read_csv(path)     
    return df

# Define the state and action spaces
state_dim = 5
action_dim = 2
hidden_dim = 10

# Initialize the QLearning agent
agent = QLearningAgent(state_dim, action_dim, hidden_dim)

#Read in training data into a data frame
#data should be in the same folder, input to function should be target file name and return file name
save_path = '/home/paulalinux20/Optimal_Control_Project/flappy-bird-gym/data/'
#file_name = 'data_d29t184122'    
file_name = 'data_train'    
path = save_path + file_name + '.csv'

df = readCSV(path) #obtain the dataset and the variables from csv file 
#x0,x1,y0,y1,v,a,r

#get initial state, action, reward
initial_data= df.iloc[0]
state = initial_data[:5]
action = int(initial_data[-2])
reward = initial_data[-1]

#skip first row
first_row = True

# train through all the rows in saved data one time
for row in df.itertuples(): #options are iterrows(), values()
    if first_row:
        first_row = False
        pass
    else:
        #index for all values
        next_state = row[:5]
        state = np.array(state).reshape(-1,state_dim)
        next_state = np.array(next_state).reshape(-1,state_dim)
        agent.update_Q(state, action, reward, next_state)
        state = row[:5]
        action = int(row[-2])
        reward = row[-1]
   
# to save    
agent.save_agent('agent_test.pkl')
# load
q_agent = QLearningAgent.load_agent('agent_test.pkl')     

#
# # Test the agent by following the greedy policy
# state = 0 # set the initial state
# done = False # set the "done" flag to False
# while not done:
#     # Choose an action using the greedy policy
#     action = agent.get_best_action(state)
#
#     # Take the action and observe the reward and new state
#     reward = -1 # a negative reward for every step
#     next_state = state + action + np.random.choice([-1, 0, 1]) # simulate a transition to a new state
#
#     # Update the current state and check if the episode is done
#     state = next_state
#     if state == 3: # if the agent reaches the goal state
#         done = True
# print("Final state:", state) # should be 3
