# MIT License
#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" Tests the simple-observations version of the Flappy Bird environment with a
random agent.
"""

import time

import sys
sys.path.insert(0, '/home/paulalinux20/Optimal_Control_Project/flappy-bird-gym')
import flappy_bird_gym


import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cvx

from flappy_bird_gym.envs.game_logic import * 

from networkQapprox import *

import csv
import pandas as pd


def show_obs(obs):
    #plt.style.use('_mpl-gallery-nogrid')
    # plot
    fig, ax = plt.subplots()
    ax.imshow(obs)
    plt.show()

def simple_control(obs):
        '''
        0 means do nothing, 1 means flap
        '''
        c = -0.05 #from paper

        if obs[1] < c:
            action = 1
        else:
            action = 0  
        
        return action

def mpc_control(obs, N):
    x0= obs[0,0]
    y0= obs[1,0]
    v0 = obs[2,0]
    Q = np.eye(1)
    R = np.eye(1)
    M = 1000 # big M parameters
    n = 1
    m = 1
    c = 10
    Y = cvx.Variable((N+1,1))
    X = cvx.Variable((N+1,1))
    V = cvx.Variable((N+1,1))
    U = cvx.Variable((N,1), boolean = True)

    cost_terms = []
    constraints = []

    constraints.append( Y[0] == y0 )
    constraints.append( V[0] == v0 )
    for k in range(N):
        # dynamics
        constraints.append( Y[k] - V[k] == Y[k+1] )
        constraints.append( X[k] + PIPE_VEL_X == X[k+1])
        constraints.append( -V[k+1] + M*U[k] <= -PLAYER_FLAP_ACC + M )
        constraints.append( V[k+1] + M*U[k] <= PLAYER_FLAP_ACC + M )
        constraints.append( -V[k+1] + V[k] - M*U[k] <= -PLAYER_ACC_Y )
        constraints.append( V[k+1] - V[k] - M*U[k] <= PLAYER_ACC_Y )
        # constraints.append( Y[k+1] <= 400)
        # constraints.append( Y[k+1] >= -400)


        constraints.append( V[k+1] <= PLAYER_MAX_VEL_Y)
        
        # cost_terms.append( cvx.quad_form(Y[k], np.eye(1)))
        # cost_terms.append( Y[k]**2 )

        cost_terms.append( U[k] )
        

    cost_terms.append( cvx.norm(Y[N],'inf') )
    # cost_terms.append( cvx.norm(V[N],'inf') )

    constraints.append( Y[N] <= 0.8*np.abs(y0))
    constraints.append( Y[N] >= -0.8*np.abs(y0))

    

    objective = cvx.Minimize( cvx.sum( cost_terms ) )

    prob = cvx.Problem(objective, constraints)
    prob.solve()

    if prob.status == 'infeasible':
        print(prob.status)
        return 0
    else:
        return U.value[0]

def plot_scene():
    plt.plot([0,BASE_WIDTH,BASE_WIDTH,0,0],
             [0,0,BASE_HEIGHT,BASE_HEIGHT,0])
    plt.plot([0,BACKGROUND_WIDTH,BACKGROUND_WIDTH,0,0],
            [0,0,BACKGROUND_HEIGHT,BACKGROUND_HEIGHT,0])
    center = [BASE_WIDTH/2,BASE_HEIGHT/2]
    center_x, center_y = center
    plt.plot
    plt.show()

def main():
    # env = gym.make("flappy_bird_gym:FlappyBird-v0")
    # env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    env = flappy_bird_gym.make("FlappyBird-v0")

    score = 0
    env._normalize_obs = False
    obs = env.reset()
    data = []

    while True:
        env.render()

        # action = simple_control(obs)
        action = mpc_control(obs, 5)

        # Processing:
        obs, reward, done, info = env.step(action)

        score += reward
        data.append((obs, action, reward))
        print(f"Obs: {obs}\n"
              f"Score: {score}\n"
              f"Info: {info}\n"
              f"Data: {data[-1]}")
        # time.sleep(1 / 30)
        # time.sleep(1 / 10)

        if done:
            env.render()
            time.sleep(0.5)
            break

    env.close()

def getState(obs):
    obs_col = len(obs[0])
    if obs_col < 2:
        pad = np.zeros((len(obs),1))
        obs = np.concatenate((obs,pad), axis=1)
    flat_obs = obs.flatten()
    state = flat_obs[0:-1]
    return flat_obs, state.reshape(-1,len(state))

def learned_control(agent, state):        
    action = agent.get_best_action(state)
    return action

def readCSV(path):
    #from this data we obtain the dataset and the variables
    df = pd.read_csv(path)     
    return df

def trainQ():
    # Define the state and action spaces
    state_dim = 5
    action_dim = 2
    hidden_dim = 10

    # Initialize the QLearning agent
    agent = QLearningAgent(state_dim, action_dim, hidden_dim)

    #Read in training data into a data frame
    #data should be in the same folder, input to function should be target file name and return file name
    save_path = '/home/paulalinux20/Optimal_Control_Project/flappy-bird-gym/data/'
    file_name = 'data_d29t184122_train'   
    shuffled = True 
    #file_name = 'data_d29t184122'   
    # shuffled = False 
    path = save_path + file_name + '.csv'

    loops = 1    

    for loop_through_data in range(loops):

        df = readCSV(path) #obtain the dataset and the variables from csv file 
        #x0,x1,y0,y1,v,a,r


        if not shuffled:

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
                        
        else:
            # train through all the rows in saved data one time
            for row in df.itertuples(): #options are iterrows(), values()
                #get values from csv file
                state = row[:5]
                action = int(row[6]) #row indecing is one off for pandas
                reward = row[7]
                next_state = row[-5:]
                #for training the network, need to pass the states in the right shape
                state = np.array(state).reshape(-1,state_dim)
                next_state = np.array(next_state).reshape(-1,state_dim)
                agent.update_Q(state, action, reward, next_state)
        
        print("training loop #" + str(loop_through_data) + " done")
        
    # to save    
    agent.save_agent('agent_more.pkl')
    # load
    q_agent = QLearningAgent.load_agent('agent_more.pkl')    
    return q_agent

def RLmain(q_agent):    

    env = flappy_bird_gym.make("FlappyBird-v0")

    score = 0
    env._normalize_obs = False
    obs = env.reset()
    flat_obs, state = getState(obs)
    data = []

    while True:
        env.render()

        # action = simple_control(obs)
        action = learned_control(q_agent, state)

        # Processing:
        obs, reward, done, info = env.step(action)

        flat_obs, state = getState(obs)
        
        #creates a row of states, action, and reward for saving
        #flat_obs is the state flattened into a vector, padded with zeros if no second pipe is visible
        #if action is an int, it must be put into a 
        data.append(np.append((np.concatenate([flat_obs[0:-1], np.array([action])])), reward))

        score += reward
        print(f"Obs: {obs}\n"
              f"Score: {score}\n"
              f"Info: {info}\n"
              f"Data: {data[-1]}")
        # time.sleep(1 / 30)
        #time.sleep(1 / 10)

        if done:
            env.render()
            time.sleep(0.5)
            break

    env.close()

if __name__ == "__main__":

    #q_agent = trainQ()
    q_agent = QLearningAgent.load_agent('agent_test.pkl')  
    
    print("trained")

    loops = 1000
    counter = 0

    while counter < loops:
        RLmain(q_agent)
        counter+=1

    #main()
    
