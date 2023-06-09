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
    env._normalize_obs = True
    obs = env.reset()
    data = []

    while True:
        env.render()

        # action = simple_control(obs)
        
        obs_in = tf.expand_dims(obs, axis=0)
        action = tf.argmax(loaded_model(obs_in)[0]).numpy()

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

        if done:
            env.render()
            time.sleep(0.5)
            break

    env.close()

if __name__ == "__main__":
    main()