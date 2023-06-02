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

class MPC_Control():
    def __init__(self, state_dim = 2, control_dim = 1, N = 10):
        self.N = N
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.path = cvx.Variable((N+1,state_dim)) # y and y_vel
        self.control = cvx.Variable((N,control_dim), boolean = True)
        self.Ys = []
        self.infeasability_count = 0
        self.control_backup = np.zeros((N,1))


    def update(self, obs):
        x0, x1, y0, y1, v0, _, y_actual, _= obs.flatten()
        N = self.N

        Q = np.eye(self.state_dim)
        R = np.eye(self.control_dim)
        M = 500

        Y_LOW = (BACKGROUND_HEIGHT - BASE_HEIGHT) - y_actual
        Y_HIGH = -y_actual
        PIPE_GAP = 100 - PLAYER_WIDTH
        PAD = PLAYER_WIDTH

        Yub = np.zeros((N+1,1))
        Ylb = np.zeros((N+1,1))
        X = np.zeros((N+1,1))

        Y = self.path[:,0]
        V = self.path[:,1]
        U = self.control

        cost_terms = []
        constraints = []

        constraints.append( Y[0] == 0 )
        constraints.append( V[0] == v0 )
        for k in range(N+1):
            # collision boundary
            X[k] = k*-PIPE_VEL_X
            print(x0 - X[k])
            if x0 - X[k] <= PIPE_WIDTH + PAD and x0 - X[k] >= - PAD:
                Yub[k] = y0 - PIPE_GAP/2
                Ylb[k] = y0 + PIPE_GAP/2
            elif x1 - X[k] <= PIPE_WIDTH + PAD and x1 - X[k] >= 0 - PAD:
                Yub[k] = y1 - PIPE_GAP/2
                Ylb[k] = y1 + PIPE_GAP/2
            else:
                Yub[k] = Y_HIGH #max([Y_HIGH, (yf - PIPE_GAP/2) - (xf - X[k] - PIPE_WIDTH)])
                Ylb[k] = Y_LOW #min([Y_LOW, (yf + PIPE_GAP/2) + (xf - X[k] - PIPE_WIDTH)])
            constraints.append( Y[k] >= Yub[k] )
            constraints.append( Y[k] <= Ylb[k] )

        for k in range(N):
            # dynamics
            constraints.append( Y[k] + V[k+1] == Y[k+1] )
            constraints.append( -V[k+1] + M*U[k] <= -(PLAYER_FLAP_ACC) + M )
            constraints.append( V[k+1] + M*U[k] <= (PLAYER_FLAP_ACC) + M )
            constraints.append( -V[k+1] + V[k] - M*U[k] <= -PLAYER_ACC_Y )
            constraints.append( V[k+1] - V[k] - M*U[k] <= PLAYER_ACC_Y )
            constraints.append( V[k+1] <= PLAYER_MAX_VEL_Y)
            cost_terms.append( U[k] )

        cost_terms.append( -Y[N] )
        
        objective = cvx.Minimize( cvx.sum( cost_terms ) )
        prob = cvx.Problem(objective, constraints)
        prob.solve(solver='GUROBI')

        if prob.status == 'infeasible' or prob.status == 'infeasible_or_unbounded':
            print(prob.status)
            self.Ys.append([y_actual, 0, Yub[0,0], Ylb[0,0]])
            if self.infeasability_count < N:
                return_u = self.control_backup[self.infeasability_count]
                self.infeasability_count += 1
            return return_u, (np.zeros((N+1,1)), Yub, Ylb)
        else:
            print(prob.status)
            self.Ys.append([y_actual, self.path.value[0,1], Yub[0,0], Ylb[0,0]])
            self.control_backup = np.copy(U.value)
            self.infeasability_count = 0
            return U.value[0], (Y.value, Yub, Ylb)


def main():
    # env = gym.make("flappy_bird_gym:FlappyBird-v0")
    # env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    env = flappy_bird_gym.make("FlappyBird-v0")
    N = 51 # for MPC
    agent = MPC_Control(2,1,N)
    score = 0
    env._normalize_obs = False
    obs = env.reset()
    data = []
    while True:
        env.render()
        # action = simple_control(obs)
        action, Ys = agent.update(obs)

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
            y = np.asarray(agent.Ys)[:,0]
            ylb = np.asarray(agent.Ys)[:,2]
            yub = np.asarray(agent.Ys)[:,3]
            plt.plot(y)
            plt.plot(ylb+y)
            plt.plot(yub+y)
            plt.gca().invert_yaxis()
            plt.show()
            break

    env.close()


if __name__ == "__main__":
    # obs = np.array([[ 77.+50., 221.+50.],
    #                 [50., 50.],
    #                 [  -2.,   -2.],
    #                 [25, 25]])
    # # obs = np.array([[ 73., 217.],
    # #                 [ -4.,  22.],
    # #                 [ -8.,  -8.]])
    # # obs = np.array([[ 89., 233.],
    # #                 [ 29., 115.],
    # #                 [  0.,   0.]])
    obs_list = [
        np.array([[ 77.+50., 221.+50.],
                [50., 50.],
                [  -2.,   -2.],
                [25, 25]])
        ]
    N = 51
    agent = MPC_Control(N = N)
    for obs in obs_list:
        agent.update(obs)
    main()
