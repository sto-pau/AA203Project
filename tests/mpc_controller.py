
import sys
sys.path.insert(0, '/home/abc/Documents/aa203/AA203Project')

import flappy_bird_gym
from flappy_bird_gym.envs.game_logic import * 
import matplotlib.pyplot as plt
import cvxpy as cvx
import numpy as np
from collections import deque

class MPC_Control():
    def __init__(self, state_dim = 2, control_dim = 1, N = 10):
        self.N = N
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.path = cvx.Variable((N+1,state_dim)) # y and y_vel
        self.control = cvx.Variable((N,control_dim), boolean = True)
        self.log = []
        self.infeasability_count = 0
        self.control_backup = np.zeros((N,1))
        self.Yub = deque(maxlen=N+1)
        self.Ylb = deque(maxlen=N+1)
        

    def get_obstacles(self, x, env):
        """ 
        Returns True if player collides with the ground (base) or a pipe.
        """
        upper_bound = 0
        lower_bound = 0
        for y in range(512):
            # if player crashes into ground
            if y + PLAYER_HEIGHT >= env._game.base_y - 1:
                lower_bound = y - 3
                break
            else:
                player_rect = pygame.Rect(x, y,
                                            PLAYER_WIDTH, PLAYER_HEIGHT)

                for up_pipe, low_pipe in zip(env._game.upper_pipes, env._game.lower_pipes):
                    # upper and lower pipe rects
                    up_pipe_rect = pygame.Rect(up_pipe['x'], up_pipe['y'],
                                                PIPE_WIDTH, PIPE_HEIGHT)
                    low_pipe_rect = pygame.Rect(low_pipe['x'], low_pipe['y'],
                                                PIPE_WIDTH, PIPE_HEIGHT)

                    # check collision
                    up_collide = player_rect.colliderect(up_pipe_rect)
                    low_collide = player_rect.colliderect(low_pipe_rect)

                    if up_collide or low_collide:
                        break
                if up_collide:
                    upper_bound = y + 3
                if low_collide:
                    lower_bound = y -3
                    break
        self.Yub.append(upper_bound)
        self.Ylb.append(lower_bound)
    

    def update(self, obs):
        y0 = obs['y']
        v0 = obs['vy']
        N = self.N
        M = 500

        PIPE_GAP = 100
        PAD = 3

        Y = self.path[:,0]
        V = self.path[:,1]
        U = self.control

        cost_terms = []
        constraints = []

        constraints.append( Y[0] == y0 )
        constraints.append( V[0] == v0 )

        for k in range(N):
            # dynamics
            constraints.append( Y[k] + V[k+1] == Y[k+1] )
            constraints.append( -V[k+1] + M*U[k] <= -(PLAYER_FLAP_ACC) + M )
            constraints.append( V[k+1] + M*U[k] <= (PLAYER_FLAP_ACC) + M )
            constraints.append( -V[k+1] + V[k] - M*U[k] <= -PLAYER_ACC_Y )
            constraints.append( V[k+1] - V[k] - M*U[k] <= PLAYER_ACC_Y )
            constraints.append( V[k+1] >= PLAYER_MIN_VEL_Y)
            constraints.append( V[k+1] <= PLAYER_MAX_VEL_Y)
            constraints.append( Y[k] <= self.Ylb[k] )
            constraints.append( Y[k] >= self.Yub[k] )
            cost_terms.append( cvx.abs(Y[k] - (self.Ylb[k] + self.Yub[k])/2) )
            cost_terms.append( cvx.abs(V[k]))
            cost_terms.append( U[k] )


        cost_terms.append( cvx.abs(V[N]) )
        cost_terms.append( cvx.abs(Y[N] + (self.Ylb[N] + self.Yub[N])/2) )

        objective = cvx.Minimize( cvx.sum( cost_terms ) )
        prob = cvx.Problem(objective, constraints)
        prob.solve(solver='GUROBI')

        if prob.status == 'infeasible' or prob.status == 'infeasible_or_unbounded':
            print(prob.status)
            if self.infeasability_count < N-1:
                self.infeasability_count += 1
                return_u = self.control_backup[self.infeasability_count]
            else:
                return_u = 0
            self.log.append([y0, v0, return_u, self.Yub[0], self.Ylb[0]])
            return return_u
        else:
            print(prob.status)
            self.control_backup = np.copy(U.value)
            self.infeasability_count = 0
            self.log.append([y0, v0, U.value[0], self.Yub[0], self.Ylb[0]])
            return U.value[0]

