import gym
import random
import numpy as np

class MultiTaskEnv(gym.Env):
    def __init__(self, env, config):
        self.env_name = env.name
        self.env = env

        # TODO: move them to config.py
        self.config = config


    @property
    def action_size(self):
        return self.env.action_space.n

    def act(self, action):
        state, reward, terminal, info = self.env.step(action)
        return self._convert_state(state), reward, terminal

    def random_new_game(self):
        self.env.reset()
        action = np.random.randint(self.env.action_space.n)
        state, reward, terminal, info = self.env.step(action)

        return state, reward, action, terminal


    def reset(self):
        state, reward, terminal = self.env.reset()
        return self._convert_state(state)

    def _convert_state(self, state):
        # TODO: Now if fills with zeros, use better convertion of different tasks
        return np.resize(state, [self.config.state_height, self.config.state_width])

