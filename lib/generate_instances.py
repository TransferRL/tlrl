import pickle as cPickcle
from env import AcrobotEnv, MountainCarEnv, ThreeDMountainCarEnv, CartPoleEnv, ThreeDCartPoleEnv, MazeEnv
import numpy as np
import deepq
import os.path
import traceback
import gym

# 2D MC -> 3D MC
# 2D MC -> 2D Cartpole
# Acrobot-v1 -> 2D Cartpole
# 2D Maze -> 3D Maze
# Breakout-ram-v0 -> Pong-ram-v0
# Pong-ram-v0 -> Breakout-ram-v0


NUM_RAND_INSTANCES = 5000
NUM_REAL_INSTANCES = 5000

RAND_MAP = {
    "2d_mountain_car": MountainCarEnv(),
    "3d_mountain_car": ThreeDMountainCarEnv(),
    "2d_cart_pole":CartPoleEnv(),
    "3d_cart_pole": ThreeDCartPoleEnv()
}

R_MAP = {
    "2d_mountain_car": MountainCarEnv(),
    "3d_mountain_car": ThreeDMountainCarEnv(),
    "2d_cart_pole":CartPoleEnv(),
    "3d_cart_pole": ThreeDCartPoleEnv(),
    "acrobot": AcrobotEnv(),
    "2d_maze": MazeEnv(maze_size=(5,5)),
    "breakout": gym.make("Breakout-ram-v0"),
    "Pong": gym.make("Pong-ram-v0")
}

SOURCE_MAP = {
    "2d_mountain_car": MountainCarEnv(),
    "acrobot": AcrobotEnv(),
    "2d_maze": MazeEnv(maze_size=(5,5)),
    "breakout": gym.make("Breakout-ram-v0"),
    "Pong": gym.make("Pong-ram-v0")
}
def get_rand_instance(env):
    env.reset()
    state = env.observation_space.sample()
    env.set_state(state)
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    return [state, action, next_state, reward, done]


def generate_optimal_samples(env, modelpath):
    act = deepq.load(modelpath)
    q_values_graph = deepq.build_graph.build_q_values(**act._act_params)

    replay_memory = []  # reset
    for ep in range(100):  # 100 episodes
        obs, done = env.reset(), False
        result = []
        while not done:
            action = act(obs[None])[0]
            q_values = q_values_graph(obs[None])
            n_obs, rew, done, _ = env.step(action)
            # print([obs, action, n_obs, rew, done, q_values[0][action]])
            result.append([obs, action, n_obs, rew, done, q_values[0][action]])
            obs = n_obs
        replay_memory.append(result)
    return replay_memory


def train_source_tasks():
    names = ["2d_mountain_car", "2d_cart_pole", "acrobot"]
    envs = [MountainCarEnv(), CartPoleEnv(), AcrobotEnv()]

    model = deepq.models.mlp([64], layer_norm=True)

    for name, env in SOURCE_MAP.items():
        file_path = "../models/"+name+"_dqn.pkl"
        if os.path.exists(file_path): continue
        act = deepq.learn(
          env,
          q_func=model,
          lr=1e-3,
          max_timesteps=40000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.1,
          print_freq=1,
          param_noise=False
          )
        act.save(file_path)

def process_random_ins():

    print("==============================")
    print("Generating random instances for ")

    names = ["2d_mountain_car", "3d_mountain_car", "2d_cart_pole", "3d_cart_pole", "acrobot"]
    envs = [MountainCarEnv(), ThreeDMountainCarEnv(), CartPoleEnv(), ThreeDCartPoleEnv(), AcrobotEnv()]
    # Generate random instances:
    for name, env in RAND_MAP.items():
        result = []
        print("Generating random samples  " + name)
        try:
            file_path = "../data/" + name + "/random_instances.pkl"
            if os.path.exists(file_path): continue
            for _ in range(NUM_RAND_INSTANCES):
                t = get_rand_instance(env)
                print(t)
                result.append(t)
            with open(file_path, "wb+") as f:
                cPickcle.dump(result, f)
        except Exception as e:
            print(e)
            traceback.print_exc()



    print("Done!")
    print("==============================")

def process_optimal_ins():

    print("==============================")
    print("Generating optimal instances...")
    # Generate optimals
    # names = ["2d_mountain_car", "acrobot"]
    # envs = [MountainCarEnv(), CartPoleEnv()]
    for name, env in SOURCE_MAP.items():
        try:
            file_path = "../data/" + name + "/optimal_instances.pkl"
            model_path = "../models/" + name + "_dqn.pkl"
            if not os.path.exists(model_path) or os.path.exists(file_path): continue
            print("Generating optimal samples  " + name)
            with open(file_path, "wb+") as f:
                cPickcle.dump(generate_optimal_samples(env, modelpath=model_path), f)
        except Exception as e:
            print(e)
            traceback.print_exc()
    print("Done!")
    print("==============================")


def process_real_ins():

    print("==============================")
    print("Generating realistic instances...")
    # Generate optimals
    # names = ["2d_mountain_car", "acrobot"]
    # envs = [MountainCarEnv(), CartPoleEnv()]
    for name, env in R_MAP.items():
        try:
            file_path = "../data/" + name + "/realistic_instances.pkl"
            if os.path.exists(file_path): continue
            print("Generating realistic samples  " + name)
            env.reset()
            results = []
            state = env.reset()
            for j in range(NUM_REAL_INSTANCES):
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)
                results.append([state, action, next_state, reward, done])
                if done:
                    state = env.reset()
                else:
                    state = next_state
            with open(file_path, "wb+") as f:
                cPickcle.dump(results, f)
        except Exception as e:
            print(e)
            traceback.print_exc()
    print("Done!")
    print("==============================")

def main():
    # train_source_tasks()

    process_random_ins()
    # process_optimal_ins()
    process_real_ins()




def test_generated():

    # Read example
    # Data format:
    # Arrays of instances. Each instance:
    # [state, action, next_state, reward, done]
    # Print random instances:
    for name, env in RAND_MAP.items():
        with open("../data/" + name + "/random_instances.pkl", "rb") as f:
            result = cPickcle.load(f)
        print("------  " + name + "   -------")
        print(result[:3])
        print("------------------------------------")


    for name, env in R_MAP.items():
        with open("../data/" + name + "/realistic_instances.pkl", "rb") as f:
            result = cPickcle.load(f)
        print("------  " + name + "   -------")
        print(result[:3])
        print("------------------------------------")

    # Read example optimals samples
    # Data format:
    # Arrays of episodes data. Each episode consists all instances.

    # Print optimals
    for name, env in SOURCE_MAP.items():
        print("------  " + name + "   -------")
        try:
            with open("../data/" + name + "/optimal_instances.pkl", "rb") as f:
                episodes = cPickcle.load(f)
                lengths = [len(episode) for episode in episodes]
            print(np.asarray(lengths).sum())
            print(episodes[-1][:10])
        except Exception as e:
            print(e)
        print("------------------------------------")

if __name__ == "__main__":
    main()
    test_generated()

