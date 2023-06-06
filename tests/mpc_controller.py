import pygame
import cvxpy as cvx
import numpy as np
from collections import deque

############################ Speed and Acceleration ############################
PIPE_VEL_X = -4

PLAYER_MAX_VEL_Y = 10  # max vel along Y, max descend speed
PLAYER_MIN_VEL_Y = -8  # min vel along Y, max ascend speed

PLAYER_ACC_Y = 1       # players downward acceleration
PLAYER_VEL_ROT = 3     # angular speed

PLAYER_FLAP_ACC = -9   # players speed on flapping
################################################################################


################################## Dimensions ##################################
PLAYER_WIDTH = 34
PLAYER_HEIGHT = 24

PIPE_WIDTH = 52
PIPE_HEIGHT = 320

BASE_WIDTH = 336
BASE_HEIGHT = 112

BACKGROUND_WIDTH = 288
BACKGROUND_HEIGHT = 512

BASE_Y = BACKGROUND_HEIGHT*0.79
################################################################################

class MPC_Control():
    '''
    initialize mpc, cvx and boundary map queue variables and parameters
    '''
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
        self.setup = True
        

    def get_obstacles(self, x, obs):
        """ 
        Returns True if player collides with the ground (base) or a pipe.
        """
        upper_pipes = obs['upper_pipes']
        lower_pipes = obs['lower_pipes']
        PAD = 1
        upper_bound = 0 
        lower_bound = 0
        for y in range(512):
            # if player crashes into ground
            if y + PLAYER_HEIGHT >= BASE_Y - 1:
                lower_bound = y - PAD
                break
            else:
                player_rect = pygame.Rect(x, y,
                                            PLAYER_WIDTH, PLAYER_HEIGHT)

                for up_pipe, low_pipe in zip(upper_pipes, lower_pipes):
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
                    upper_bound = y + PAD
                if low_collide:
                    lower_bound = y - PAD
                    break
        self.Yub.append(upper_bound)
        self.Ylb.append(lower_bound)
    
    def update_map(self, obs):
        if self.setup == True:
            for x in np.arange(obs['x'],
                               obs['x'] + -self.N*PIPE_VEL_X,
                               -PIPE_VEL_X):
                self.get_obstacles(x, obs)
            self.setup == False
        self.get_obstacles(obs['x'] + -self.N*PIPE_VEL_X,obs)

    def update_control(self, obs):
        '''
        update controls using new observations
        '''
        y0 = obs['y']
        v0 = obs['vy']
        N = self.N
        M = 500

        Y = self.path[:,0]
        V = self.path[:,1]
        U = self.control

        cost_terms = []
        constraints = []

        # IC constraints
        constraints.append( Y[0] == y0 )
        constraints.append( V[0] == v0 )

        for k in range(N):

            # dynamics and controls constraints
            constraints.append( Y[k] + V[k+1] == Y[k+1] )
            constraints.append( -V[k+1] + M*U[k] <= -(PLAYER_FLAP_ACC) + M )
            constraints.append( V[k+1] + M*U[k] <= (PLAYER_FLAP_ACC) + M )
            constraints.append( -V[k+1] + V[k] - M*U[k] <= -PLAYER_ACC_Y )
            constraints.append( V[k+1] - V[k] - M*U[k] <= PLAYER_ACC_Y )

            # boundary constraints
            constraints.append( Y[k] <= self.Ylb[k] )
            constraints.append( Y[k] >= self.Yub[k] )

            # cost for distance from center, none zero y velocity and control
            cost_terms.append( cvx.abs(Y[k] - (self.Ylb[k] + self.Yub[k])/2) )
            cost_terms.append( cvx.abs(V[k]))
            cost_terms.append( U[k] )

        # cost for last term
        cost_terms.append( cvx.abs(V[N]) ) 
        cost_terms.append( cvx.abs(Y[N] + (self.Ylb[N] + self.Yub[N])/2) )

        # solve 
        objective = cvx.Minimize( cvx.sum( cost_terms ) )
        prob = cvx.Problem(objective, constraints)
        prob.solve(solver='GUROBI')

        # check infeasibility and return last cashed control if it is
        if prob.status == 'infeasible' or prob.status == 'infeasible_or_unbounded':
            print(prob.status)
            if self.infeasability_count < N-1:
                self.infeasability_count += 1
                return_u = self.control_backup[self.infeasability_count]
            else:
                return_u = 0
            self.log.append([y0, v0, return_u, self.Yub[0], self.Ylb[0]])
            return return_u
        # otherwise update controls and return next action
        else:
            print(prob.status)
            self.control_backup = np.copy(U.value)
            self.infeasability_count = 0
            self.log.append([y0, v0, U.value[0], self.Yub[0], self.Ylb[0]])
            return U.value[0]

