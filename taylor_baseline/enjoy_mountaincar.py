import gym

import deepq

from lib.env.threedmountain_car import ThreeDMountainCarEnv

def main():
    # env = gym.make("MountainCar-v0")
    env = ThreeDMountainCarEnv()
    act = deepq.load("mountaincar_model_working.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            # env.render()
            env.render_orthographic()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            print(act(obs[None])[0])
            print(obs)
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
