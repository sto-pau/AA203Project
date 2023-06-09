import gym
import sys
sys.path.insert(0, '~/AA203/AA203Project/flappy_bird_gym')
import flappy_bird_gym

from gym.wrappers import Monitor
from collections import deque
import tensorflow as tf
import numpy as np
import random
import math
import time
import os
save_path = os.path.expanduser('~/AA203/AA203Project/tests/model1')

loaded_model = tf.keras.models.load_model(save_path)

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

def main():
    env = flappy_bird_gym.make("FlappyBird-v0")
    score = 0
    env._normalize_obs = False
    obs = env.reset()
    data = []

    UPDATED_Y = 200 # try all values in range [0,404]
    UPDATED_X = 60.0  # anything <= 90
    env._game.player_y = UPDATED_Y
    for up_pipe, low_pipe in zip(env._game.upper_pipes, env._game.lower_pipes):
        up_pipe['x'] += UPDATED_X*-4
        low_pipe['x'] += UPDATED_X*-4
    # env._game.upper_pipes[0]['y'] = int(env._game.base_y * 0.2 - 320)
    # env._game.lower_pipes[0]['y'] = int(env._game.base_y * 0.2 + env._game._pipe_gap_size)
    # env._game.upper_pipes[1]['y'] = int(env._game.base_y * 0.8 - 320 - env._game._pipe_gap_size)
    # env._game.lower_pipes[1]['y'] = int(env._game.base_y * 0.8 )

    while True:
        
        env.render()

        # action = simple_control(obs)
        
        obs_in = tf.expand_dims(obs, axis=0)
        action = tf.argmax(loaded_model(obs_in)[0]).numpy()
        action = simple_control(obs)
        # obs_in = tf.expand_dims(obs, axis=0)
        # action = tf.argmax(loaded_model(obs_in)[0]).numpy()

        # Processing:
        obs, reward, done, info = env.step(action)

        score += reward
        data.append((obs, action, reward))
        print(f"Obs: {obs}\n"
              f"Score: {score}\n"
              f"Info: {info}\n"
              f"Done: {done}\n"
              # f"Data: {data[-1]}"
              )
        time.sleep(1/100)

        if done:
            env.render()
            time.sleep(0.5)
            break

    env.close()

if __name__ == "__main__":
    main()