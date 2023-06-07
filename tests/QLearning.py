import gym
import flappy_bird_gym
from gym.wrappers import Monitor
from collections import deque
import tensorflow as tf
import numpy as np
import random
import time
# import math
# import glob
# import io
# import base64
import os
save_path = os.path.expanduser('~/AA203/AA203Project/tests/model11')

# Load gym environment and get action and state spaces.
env = flappy_bird_gym.make("FlappyBird-v0")
num_features = env.observation_space.shape[0]
num_actions = env.action_space.n
print('Number of state features: {}'.format(num_features))
print('Number of possible actions: {}'.format(num_actions))

class DQN(tf.keras.Model):
  """Dense neural network class."""
  def __init__(self):
    super(DQN, self).__init__()
    self.dense1 = tf.keras.layers.Dense(32, activation="relu")
    self.dense2 = tf.keras.layers.Dense(32, activation="relu")
    self.dense3 = tf.keras.layers.Dense(num_actions, dtype=tf.float32) # No activation
    
  def call(self, x):
    """Forward pass."""
    x = self.dense1(x)
    x = self.dense2(x)
    return self.dense3(x)

if os.path.exists(save_path):
    # Load the saved model
    print("Existing network weights being used")
    main_nn = tf.keras.models.load_model(save_path)
    main_nn.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    target_nn = tf.keras.models.load_model(save_path)
    target_nn.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
else:
    # Create a new model
    print("Creating new models")
    main_nn = DQN()
    target_nn = DQN()
    # Train the model or perform any other operations

optimizer = tf.keras.optimizers.Adam(1e-4) #Optimizer for minimizing the loss function
mse = tf.keras.losses.MeanSquaredError() #Type of Loss function

#an object of 'ReplayBuffer' class stores the data
class ReplayBuffer(object):
  """Experience replay buffer that samples uniformly."""
  def __init__(self, size):
    self.buffer = deque(maxlen=size)

  def add(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def __len__(self):
    return len(self.buffer)

  def sample(self, num_samples):
    states, actions, rewards, next_states, dones = [], [], [], [], []

    #randomly select 'num_samples' number of indices starting from 0 to len(self.buffer)
    idx = np.random.choice(len(self.buffer), num_samples)
    for i in idx:
      elem = self.buffer[i]
      state, action, reward, next_state, done = elem
      states.append(np.array(state, copy=False))
      actions.append(np.array(action, copy=False))
      rewards.append(reward)
      next_states.append(np.array(next_state, copy=False))
      dones.append(done)
    
    #Converting lists to np arrays
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards, dtype=np.float32)
    next_states = np.array(next_states)
    dones = np.array(dones, dtype=np.float32)
    return states, actions, rewards, next_states, dones
   
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

def select_epsilon_greedy_action(state, epsilon, randprob):
  """Take random action with probability epsilon, else take best action."""
  result = tf.random.uniform((1,))
  if result < epsilon:
    # if np.random.choice(list(range(randprob))) == 0:
    #   return env.action_space.sample() #random
    # else:
      return simple_control(state)
  else:
    state_in = tf.expand_dims(state, axis=0) #expanding the state dims along the first axis
    return tf.argmax(main_nn(state_in)[0]).numpy() # Greedy action for state.
  
@tf.function
def train_step(states, actions, rewards, next_states, dones, discount):
  """Perform a training iteration on a batch of data sampled from the experience
  replay buffer."""
  # Calculate targets.
  next_qs = target_nn(next_states)
  max_next_qs = tf.reduce_max(next_qs, axis=-1)
  target = rewards + (1. - dones) * discount * max_next_qs
  #Gradient calculation
  with tf.GradientTape() as tape:
    qs = main_nn(states)
    action_masks = tf.one_hot(actions, num_actions)
    masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
    loss = mse(target, masked_qs)
  grads = tape.gradient(loss, main_nn.trainable_variables)
  optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))
  return loss

def main():
  # Hyperparameters.
  num_episodes = 3000
  epsilon = 1.0
  batch_size = 32
  discount = 0.99
  randprob=100 #the random action has a 1/c probability to get picked
  negrew=-3 #large negative reward. Can be tuned
  freq=1000 #Frequency of updating target neural network i.e. target_nn, with weights from the main_nn
  buffer = ReplayBuffer(100000)
  # Start training. Play game once and then train with a batch.
  last_100_ep_rewards = []
  cur_frame = 0
  print(f"Hyperparameter details:\n"
        f"model number: 10\n"
        f"num_episodes: {num_episodes}\n"
        f"discount factor: {discount}\n"
        # f"Probability of taking random action in training: {1.0/randprob}\n"
        f"Negative Reward for done=1: {negrew}\n"
        f"Batch size for training: {batch_size}\n"
        f"Frequency (in episodes) of updating target_nn: {freq}\n"
        )
  for episode in range(num_episodes+1):
    state = env.reset()
    ep_reward, done = 0, False
    while not done: #Each episode continues until the bird crashes i.e. done =1
      #'done' specifies if the game is done or still in action
      action = select_epsilon_greedy_action(state, epsilon, randprob)
      next_state, reward, done, info = env.step(action)
      #Can we modify the 'reward' and make it a function of 'done'
      reward=reward+done*negrew #penalizing done=1 (True) i.e. game finished.
      ep_reward += reward
      # Save to experience replay.
      buffer.add(state, action, reward, next_state, done)
      state = next_state
      cur_frame += 1

      # Copy main_nn weights to target_nn at multiples of freq.
      if cur_frame % freq == 0:
        target_nn.set_weights(main_nn.get_weights())

      # Train neural network.
      if len(buffer) >= batch_size:
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        loss = train_step(states, actions, rewards, next_states, dones, discount)
    
    if episode < num_episodes and epsilon > 0.1:
      epsilon -= 1/num_episodes

    # Once the length of last_... rewards reaches 100 it starts removing the oldest reward everytime a new reward comes in
    if len(last_100_ep_rewards) == 100:
      last_100_ep_rewards = last_100_ep_rewards[1:]
    last_100_ep_rewards.append(ep_reward)
      
    if episode % 50 == 0:
      print(f'Episode {episode}/{num_episodes}. Epsilon: {epsilon:.3f}. '
            f'Mean reward for last 100 episodes: {np.mean(last_100_ep_rewards):.3f}')
  main_nn.save(save_path)
  env.close()

if __name__ == "__main__":
  start_time = time.time()
  main()
  end_time = time.time()
  elapsed_time = end_time - start_time
  print("Elapsed time for training: ", elapsed_time, " seconds")
  
