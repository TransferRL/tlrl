import gym

from baselines import deepq
from lib.env.threedcartpole import ThreeDCartPoleEnv


def main():
    # env = gym.make("CartPole-v0")
    env = ThreeDCartPoleEnv()
    act = deepq.load("cartpole_model.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            # env.render()
            env.render_orthographic()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
