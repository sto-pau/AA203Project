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
sys.path.insert(0, '/home/abc/Documents/aa203/AA203Project')

import flappy_bird_gym
from flappy_bird_gym.envs.game_logic import * 
import matplotlib.pyplot as plt
import cvxpy as cvx
import numpy as np
from collections import deque

# from simple_controller import simple_control
from mpc_controller import MPC_Control




def main():
    env = flappy_bird_gym.make("FlappyBird-v0")
    env.normalize_obs = False
    obs = env.reset()

    N = 15 # for MPC
    agent = MPC_Control(2,1,N)
    for x in np.arange(obs['x'],obs['x'] + -N*PIPE_VEL_X,-PIPE_VEL_X):
        agent.get_obstacles(x, env)

    data = []
    score = 0
    while True:
        env.render()
        # action = simple_control(obs)
        agent.get_obstacles(obs['x'] + -N*PIPE_VEL_X,env)
        action = agent.update(obs)

        # Processing:
        obs, reward, done, info = env.step(action)

        score += reward
        data.append((obs, action, reward))
        print(f"Obs: {obs}\n"
            f"Score: {score}\n"
            f"Info: {info}")
        # time.sleep(1 / 30)
        # time.sleep(1 / 5)

        if done:
            env.render()
            time.sleep(0.5)
            y = np.asarray(agent.log)[:,0]
            ylb = np.asarray(agent.log)[:,3]
            yub = np.asarray(agent.log)[:,4]
            plt.plot(y)
            plt.plot(ylb)
            plt.plot(yub)
            plt.gca().invert_yaxis()
            plt.show()
            break

    env.close()


if __name__ == "__main__":
    main()
