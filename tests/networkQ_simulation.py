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
import datetime
import csv

import sys
sys.path.insert(0, '/home/abc/Documents/aa203/AA203Project')
import flappy_bird_gym


import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cvx

from networkQapprox import QLearningAgent

from flappy_bird_gym.envs.game_logic import * 


PATH = 'AA203Project/tests/agent.pkl' # path/to/agent
SAVE_DATA = True
SIMPLE = False

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
        c = 288 * -0.09 #from paper -0.05

        if obs[1][0] < c:
            action = 1
        else:
            action = 0  
        
        return np.array([action])

def main(filename):
    # env = gym.make("flappy_bird_gym:FlappyBird-v0")
    # env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    env = flappy_bird_gym.make("FlappyBird-v0")
    if SIMPLE == False: 
        agent = QLearningAgent(0,0,0)
        agent.load_agent(PATH)
    score = 0
    env._normalize_obs = False
    obs = env.reset()
    data = []
    while True:
        env.render()

        obs_col = len(obs[0])
        if obs_col < 2:
            pad = np.zeros((len(obs),1))
            obs = np.concatenate((obs,pad), axis=1)
        flat_obs = obs.flatten()

        if SIMPLE == True:
            action = simple_control(obs)
        else: 
            action = agent.get_best_action(flat_obs)

        # Processing:
        obs, reward, done, info = env.step(action)

        score += reward

        data.append(np.append((np.concatenate([flat_obs[0:-1], action])), reward))

        score += reward

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

    if info['score'] > 50 and SAVE_DATA == True:
        print("adding to csv")

        # Open the CSV file in write mode
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)

            # Write each row of data to the CSV file
            for row in data:
                writer.writerow(row)


if __name__ == "__main__":
    loops = 1000
    counter = 0

    # Get the current date and time
    current_datetime = datetime.datetime.now()
    datetime_str = current_datetime.strftime("d%dt%H%M%S")

    # Create the file name with the date and time
    filename = f"data_{datetime_str}.csv"

    while counter < loops:
        main(filename)
        counter+=1
