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
        c = -0.1*512 #from paper

        if obs[1][0] < c:
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

    R = np.eye(1)

    Yub = np.zeros((N+1,1))-y0
    Ylb = np.zeros((N+1,1))-y0
    yf = np.zeros((N+1,2))

    X = np.zeros((N+1,1))
    A = np.array([  [1, 1],
                    [0, 1]])
    # B = np.array([0, PLAYER_FLAP_ACC])
    B = np.array([0, -2])
    B = np.reshape(B, (2,1))
    C = np.array([0, PLAYER_ACC_Y])

    Y = cvx.Variable((N+1,2))
    U = cvx.Variable((N,1))

    cost_terms = []
    constraints = []

    PAD = 25
    for k in range(N+1):
        # collision boundary
        X[k] = k*-PIPE_VEL_X
        if (x0 + PAD) - X[k] < 0:
            yf[k][0] = y1
        else:
            yf[k][0] = y0

    constraints.append( Y[0] == [0, v0] )
    for k in range(N):
        # dynamics
        constraints.append( Y[k+1] == A@Y[k] + B@U[k] + C )
        constraints.append( U[k] <= 1 )
        constraints.append( U[k] >= 0 )
        constraints.append( Y[k][1] >= -9 )
        constraints.append( Y[k][1] <= 10 )
        cost_terms.append( cvx.quad_form(U[k], R*0.1) )
        cost_terms.append( cvx.norm(Y[k]-yf[k]) )

    cost_terms.append( cvx.norm(Y[N]-yf[N]) )

    objective = cvx.Minimize( cvx.sum( cost_terms ) )
    prob = cvx.Problem(objective, constraints)
    prob.solve()

    if prob.status == 'infeasible':
        print(prob.status)
        return 0, (np.zeros((N+1,1)), Yub, Ylb)
    else:
        # plt.plot(X, yf[:,0])
        # plt.plot(0,0, marker = 'x')
        # plt.plot(X, Y.value[:,0])
        # plt.gca().invert_yaxis()
        # plt.show()
        return U.value[0], (Y.value, Yub, Ylb)

def plot_scene(env, obs, Y, N):
    plt.clf
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
        a, Ys = mpc_control(obs, N)
        # action = np.random.rand(1).round()[0]
        # if obs[0][0] < 70:
        #     plot_scene(env, obs, Ys, N)
        if a != 0 :
            action = 1
        # Processing:
        obs, reward, done, info = env.step(a, action)


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
    # obs = np.array([[ 77., 221.],
    #                 [130.-200, 124.-200],
    #                 [  8.,   8.]])
    # # obs = np.array([[ 73., 217.],
    # #                 [ -4.,  22.],
    # #                 [ -8.,  -8.]])
    # # obs = np.array([[ 89., 233.],
    # #                 [ 29., 115.],
    # #                 [  0.,   0.]])
    # # obs = np.array([[477., 477.],
    # #                 [-35., -35.],
    # #                 [ -9.,  -9.]])
    # # obs = np.array([[ 77., 221.],
    # #    [-52., -13.],
    # #    [ -9.,  -9.]])
    # N = 20
    # mpc_control(obs, N)
    main()
