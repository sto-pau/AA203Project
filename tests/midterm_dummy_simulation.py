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
        c = -0.05 #from paper

        if obs[1] < c:
            action = 1
        else:
            action = 0  
        
        return action

def mpc_control(obs, N):
    x0 = obs[0,0]
    x1 = obs[0,1]
    y0 = obs[1,0]
    y1 = obs[1,1]
    v0 = obs[2,0]

    Q = np.eye(1)
    R = np.eye(1)
    M = 1000 # big M parameters
    n = 1
    m = 1

    Y_LOW = max([y0,y1]) + 100
    Y_HIGH = min([y0,y1]) - 100
    PIPE_GAP = 100

    Yub = np.zeros((N+1,1))
    Ylb = np.zeros((N+1,1))
    X = np.zeros((N+1,1))

    Y = cvx.Variable((N+1,1))
    V = cvx.Variable((N+1,1))
    U = cvx.Variable((N,1), boolean = True)

    cost_terms = []
    constraints = []

    constraints.append( Y[0] == 0 )
    constraints.append( V[0] == v0 )
    for k in range(N+1):
        # parameters
        X[k] = k*-PIPE_VEL_X
        if x0 - X[k] <= PIPE_WIDTH and x0 - X[k] >= 0:
            Yub[k] = y0 - PIPE_GAP/2
            Ylb[k] = y0 + PIPE_GAP/2
        elif x1 - X[k] <= PIPE_WIDTH and x1 - X[k] >= 0:
            Yub[k] = y1 - PIPE_GAP/2
            Ylb[k] = y1 + PIPE_GAP/2
        else:
            Yub[k] = Y_HIGH
            Ylb[k] = Y_LOW
        constraints.append( Y[k] >= Yub[k] )
        constraints.append( Y[k] <= Ylb[k] )

    for k in range(N):
        # dynamics
        constraints.append( Y[k] + V[k] == Y[k+1] )
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
        # cost_terms.append( cvx.norm(Y[k],'inf') )

        
    # cost_terms.append( U[N] )
    # cost_terms.append( cvx.norm(Y[N],'inf') )
    # cost_terms.append( cvx.norm(V[N],'inf') )

    # constraints.append( Y[N] <= 0.8*np.abs(y0))
    # constraints.append( Y[N] >= -0.8*np.abs(y0))

    

    objective = cvx.Minimize( cvx.sum( cost_terms ) )

    prob = cvx.Problem(objective, constraints)
    prob.solve()

    if prob.status == 'infeasible':
        print(prob.status)
        return 0, (np.zeros((N+1,1)), Yub, Ylb)
    else:
        return U.value[0], (Y.value, Yub, Ylb)

def plot_scene(env, obs, Y, N):
    screen_width, screen_height = env._screen_size
    # plt.plot([0,screen_width,screen_width,0,0],
    #          y-[0,0,-screen_height,-screen_height,0])
    x = env._game.player_x
    y = env._game.player_y
    plt.plot(-x,y, marker = '+')
    plt.plot(-x+screen_width,y-screen_height, marker = '+')
    
    plt.plot(0,0, marker = 'o')
    for ob in obs.T:
        p_x = ob[0]
        p_y = -ob[1]
        p_xs = [-PIPE_WIDTH+p_x, p_x]
        p_ly = [p_y - env._pipe_gap/2, p_y - env._pipe_gap/2]
        p_uy = [p_y + env._pipe_gap/2, p_y + env._pipe_gap/2]
        plt.plot(p_xs, p_ly, marker = '.')
        plt.plot(p_xs, p_uy, marker = '.')

    x_move = np.arange(0, -PIPE_VEL_X*(N+1), -PIPE_VEL_X)
    plt.plot(x_move, Y[0])
    plt.plot(x_move, -Y[2])
    plt.plot(x_move, -Y[1])
    plt.show()



def main():
    # env = gym.make("flappy_bird_gym:FlappyBird-v0")
    # env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    env = flappy_bird_gym.make("FlappyBird-v0")

    score = 0
    env._normalize_obs = False
    obs = env.reset()
    data = []
    N = 15 # for MPC
    while True:
        env.render()

        # action = simple_control(obs)
        action, Ys = mpc_control(obs, N)
        if action == 1:
            plot_scene(env, obs, Ys, N)

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


if __name__ == "__main__":
    main()
