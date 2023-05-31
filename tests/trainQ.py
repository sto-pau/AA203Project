import numpy as np
from networkQapprox import *
import csv

# Define the state and action spaces

state_dim = 5
action_dim = 2
hidden_dim = 10

# Initialize the QLearning agent
agent = QLearningAgent(state_dim, action_dim, hidden_dim)

train_steps = 
# intialize state
state = np.array([1.2, 3.4, -4]).reshape(-1,3)

def action_space(action,c1,c2):
    act = [45, -45, 12][action]
    return np.array([np.sin(act)*c1,c1+np.random.choice([-c2,c2]),  np.cos(act)+c2]).reshape(-1,3)

# train
for i in range(train_steps):
    # Choose an action using the epsilon-greedy policy
    action = agent.get_training_action(state)
    # Take the action and observe the reward and new state
    reward = np.sum(state**2 - state/2) + np.random.rand()
    print(state[0],action)
    next_state = action_space(action,np.random.rand(),np.random.rand())
    # Update the Q-value for the current state-action pair
    agent.update_Q(state, action, reward, next_state)
    # Update the current state and check if the episode is done
    state = next_state
    agent.epsilon = agent.epsilon * 0.999
    
# to save    
agent.save_agent('agent.pkl')
# load
q_agent = QLearningAgent.load_agent('agent.pkl')
     

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
