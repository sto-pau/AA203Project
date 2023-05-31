import numpy as np
import random
import tensorflow as tf
import pickle 

import os

class QNetwork:
    def __init__(self, state_dim, action_dim, hidden_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.state_dim,)),
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def predict(self, state):
        return self.model.predict(state)

    def update(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0)
        
    def save_model(self, path):
        self.model.save(path,save_format="tf")

    @classmethod
    def load_model(cls, path):
        return tf.keras.models.load_model(path)

class QLearningAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, alpha=0.1, gamma=0.99, epsilon=0.3,Q=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = QNetwork(state_dim, action_dim, hidden_dim) if Q==None else Q

    def get_training_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            q_values = self.Q.predict(state)
            return np.argmax(q_values)

    def get_best_action(self, state):
        q_values = self.Q.predict(state)
        return np.argmax(q_values)

    def update_Q(self, state, action, reward, next_state):
        q_values = self.Q.predict(state)
        next_q_values = self.Q.predict(next_state)
        td_target = reward + self.gamma * np.max(next_q_values)
        td_error = td_target - q_values[0][action]
        target = q_values
        target[0][action] += self.alpha * td_error
        self.Q.update(state, target)

    # def train(self, env, num_episodes, max_steps):
    #     for episode in range(num_episodes):
    #         state = env.reset()
    #         for step in range(max_steps):
    #             action = self.get_action(state)
    #             next_state, reward, done, _ = env.step(action)
    #             self.update_Q(state, action, reward, next_state)
    #             state = next_state
    #             if done:
    #                 break
    #         # Decrease epsilon as we progress
    #         self.epsilon = self.epsilon * 0.99
            
    def save_agent(self, path):
            q_network_path = path + '_q_network'
            self.Q.save_model(q_network_path)
            agent_dict = {'state_dim': self.state_dim,
                          'action_dim': self.action_dim,
                          'hidden_dim': self.hidden_dim,
                          'alpha': self.alpha,
                          'gamma': self.gamma,
                          'epsilon': self.epsilon,
                          'q_network_path': q_network_path}
            with open(path, 'wb') as f:
                pickle.dump(agent_dict, f)
    
    @classmethod
    def load_agent(cls, path):
            with open(path, 'rb') as f:
                agent_dict = pickle.load(f)
            q_network_path = agent_dict['q_network_path']
            q_network = QNetwork.load_model(q_network_path)
            return cls(state_dim=agent_dict['state_dim'],
                        action_dim=agent_dict['action_dim'],
                        hidden_dim=agent_dict['hidden_dim'],
                        alpha=agent_dict['alpha'],
                        gamma=agent_dict['gamma'],
                        epsilon=agent_dict['epsilon'],
                        Q=q_network)

