import gym

import sys
sys.path.insert(0, '/home/paulalinux20/Optimal_Control_Project/flappy-bird-gym')
import flappy_bird_gym

from gym.wrappers import Monitor
import collections
from collections import deque
import tensorflow as tf
import numpy as np
import random
import math
import time
import glob
import io
import base64
import os
save_path = os.path.expanduser('~/Optimal_Control_Project/flappy-bird-gym/tests/model')

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

import tqdm

import statistics

# Just some initial setup and imports
import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
import time
import numpy as np
import math
import gym

# Load gym environment and get action and state spaces.
env = flappy_bird_gym.make("FlappyBird-v0")

env._normalize_obs = True

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

def select_epsilon_greedy_action(state, epsilon):
  """Take random action with probability epsilon, else take best action."""
  result = tf.random.uniform((1,))
  state_in = tf.expand_dims(state, axis=0)
  if result < epsilon:
    return simple_control(state) # Random action (left or right).
  else:
    return tf.argmax(model(state_in)[0]).numpy() # Greedy action for state.

# This helper is for later. 
def to_onehot(size,value):
  my_onehot = np.zeros((size))
  my_onehot[value] = 1.0
  return my_onehot

OBSERVATION_SPACE = env.observation_space.shape[0] #env.observation_space.n
ACTION_SPACE = env.action_space.n

# Assume gridworld is always square
OBS_SQR= int(math.sqrt(OBSERVATION_SPACE))
STATEGRID = np.zeros((OBS_SQR,OBS_SQR))

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

actor_model = Sequential()
actor_model.add(Dense(164, kernel_initializer='lecun_uniform', input_shape=(OBSERVATION_SPACE,)))
actor_model.add(Activation('relu'))

actor_model.add(Dense(150, kernel_initializer='lecun_uniform'))
actor_model.add(Activation('relu'))

actor_model.add(Dense(ACTION_SPACE, kernel_initializer='lecun_uniform'))
actor_model.add(Activation('linear'))

a_optimizer = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
actor_model.compile(loss='mse', optimizer=a_optimizer)

critic_model = Sequential()
critic_model.add(Dense(164, kernel_initializer='lecun_uniform', input_shape=(OBSERVATION_SPACE,)))
critic_model.add(Activation('relu'))

critic_model.add(Dense(150, kernel_initializer='lecun_uniform'))
critic_model.add(Activation('relu'))

critic_model.add(Dense(1, kernel_initializer='lecun_uniform'))
critic_model.add(Activation('linear'))

c_optimizer = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
critic_model.compile(loss='mse', optimizer=c_optimizer)

import random
import time

def trainer(epochs=1000, batchSize=40, 
            gamma=0.975, epsilon=1, min_epsilon=0.1,
            buffer=80):
    
    wins = 0
    losses = 0
    # Replay buffers
    actor_replay = []
    critic_replay = []
    
    for i in range(epochs):

        observation = env.reset()
        done = False
        reward = 0
        info = None
        move_counter = 0

        while(not done):
            # Get original state, original reward, and critic's value for this state.
            #orig_state = to_onehot(OBSERVATION_SPACE,observation)
            orig_state = tf.one_hot(observation, OBSERVATION_SPACE)
            orig_reward = reward
            orig_val = critic_model(orig_state)

            if (random.random() < epsilon): #choose random action
                action = simple_control(observation)
            else: #choose best action from Q(s,a) values
                qval = actor_model(orig_state)
                action = (np.argmax(qval))
                
            #Take action, observe new state S'
            new_observation, new_reward, done, info = env.step(action)
            new_state = tf.one_hot(new_observation, OBSERVATION_SPACE)
            # Critic's value for this new state.
            new_val = critic_model(new_state)
            
            if not done: # Non-terminal state.
                target = orig_reward + ( gamma * new_val)
            else:
                # In terminal states, the environment tells us
                # the value directly.
                target = orig_reward + ( gamma * new_reward )
            
            # For our critic, we select the best/highest value.. The
            # value for this state is based on if the agent selected
            # the best possible moves from this state forward.
            # 
            # BTW, we discount an original value provided by the
            # value network, to handle cases where its spitting
            # out unreasonably high values.. naturally decaying
            # these values to something reasonable.
            best_val = max((orig_val*gamma), target)

            # Now append this to our critic replay buffer.
            critic_replay.append([orig_state,best_val])
            # If we are in a terminal state, append a replay for it also.
            if done:
                critic_replay.append( [new_state, float(new_reward)] )
            
            # Build the update for the Actor. The actor is updated
            # by using the difference of the value the critic
            # placed on the old state vs. the value the critic
            # places on the new state.. encouraging the actor
            # to move into more valuable states.
            actor_delta = new_val - orig_val                
            actor_replay.append([orig_state, action, actor_delta])
                    
            # Critic Replays...
            while(len(critic_replay) > buffer): # Trim replay buffer
                critic_replay.pop(0)
            # Start training when we have enough samples.
            if(len(critic_replay) >= buffer):
                minibatch = random.sample(critic_replay, batchSize)
                X_train = []
                y_train = []
                for memory in minibatch:
                    m_state, m_value = memory
                    y = np.empty([1])
                    y[0] = m_value
                    X_train.append(m_state.reshape((OBSERVATION_SPACE,)))
                    y_train.append(y.reshape((1,)))
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                critic_model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=0)
            
            # Actor Replays...
            while(len(actor_replay) > buffer):
                actor_replay.pop(0)                
            if(len(actor_replay) >= buffer):
                X_train = []
                y_train = []
                minibatch = random.sample(actor_replay, batchSize)
                for memory in minibatch:
                    m_orig_state, m_action, m_value = memory
                    old_qval = actor_model.predict( m_orig_state.reshape(1,OBSERVATION_SPACE,) )
                    y = np.zeros(( 1, ACTION_SPACE ))
                    y[:] = old_qval[:]
                    y[0][m_action] = m_value
                    X_train.append(m_orig_state.reshape((OBSERVATION_SPACE,)))
                    y_train.append(y.reshape((ACTION_SPACE,)))
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                actor_model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=0)

            # Bookkeeping at the end of the turn.
            observation = new_observation
            reward = new_reward
            move_counter+=1
            if done:
                if new_reward > 0 : # Win
                    wins += 1
                else: # Loss
                    losses += 1
        # Finised Epoch
        print("Game #: %s" % (i,))
        print("Moves this round %s" % move_counter)
        print("Final Position:")
        env.render()
        print("Wins/Losses %s/%s" % (wins, losses))
        if epsilon > min_epsilon:
            epsilon -= (1/epochs)

trainer()

# actor_model.save(save_path)
env.close()
