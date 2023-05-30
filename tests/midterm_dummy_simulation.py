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

        action = simple_control(obs)
        #action = mpc_control(obs, 5)

        # Processing:
        obs, reward, done, info = env.step(action)

        score += reward

        obs_col = len(obs[0])
        if obs_col < 2:
            pad = np.zeros((len(obs),1))
            obs = np.concatenate((obs,pad), axis=1)
        flat_obs = obs.flatten()

        print(f"Obs: {obs}\n"
            f"action{action}\n"
            f"reward{reward}\n")
        
        #data.append(np.concatenate([flat_obs[0:-1], action]))
        data.append(np.append((np.concatenate([flat_obs[0:-1], action])), reward))
        #flattened_data = np.concatenate([np.array(data[-1][0]).flatten(), np.array(data[-1][1]).flatten(), np.array(data[-1][2])])
        #flattened = data[-1].flatten()

        print(f"Obs: {obs}\n"
              f"Score: {score}\n"
              f"Info: {info}\n"
              f"Data: {data[-1]}\n\n")
        # time.sleep(1 / 30)
        #time.sleep(1 / 10)


        if done:
            env.render()
            time.sleep(0.5)
            break

    env.close()


if __name__ == "__main__":
    main()
