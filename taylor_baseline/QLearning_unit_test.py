import unittest
import lib.qlearning as ql
import numpy as np
import gym
from lib.env.threedmountain_car import ThreeDMountainCarEnv
from lib.env.mountain_car import MountainCarEnv
from lib.env.threedcartpole import ThreeDCartPoleEnv
from lib.env.cartpole import  CartPoleEnv
from lib.env.pendulum import PendulumEnv
# from lib.env.atari_env import AtariEnv
from lib.env.acrobot import AcrobotEnv

class MyTestCase(unittest.TestCase):
    def test_qlearning(self):
        # env = gym.envs.make("MountainCar-v0")
        # env = lib.env.mountain_car.MountainCarEnv()
        # env = ThreeDMountainCarEnv()
        # env = MountainCarEnv()
        env = CartPoleEnv()
        # env = AtariEnv()

        print(env.action_space.n)

        qlearning = ql.QLearning(env, rendering=True)
        qlearning.learn(num_episodes=100)
        dsource = qlearning.play()

        # np.savez('dsource_qlearn_3d.npz', dsource=dsource)

        assert(True)


if __name__ == '__main__':
    unittest.main()
