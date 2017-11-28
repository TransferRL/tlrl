import pickle as cPickcle
from env import AcrobotEnv, MountainCarEnv, ThreeDMountainCarEnv, CartPoleEnv, ThreeDCartPoleEnv
import numpy as np
import deepq
import os.path
import traceback

# 2D MC -> 3D MC
# 2D MC -> 2D Cartpole
# Acrobot-v1 -> 2D Cartpole
# 2D Maze -> 3D Maze
# Breakout-ram-v0 -> Pong-ram-v0
# Pong-ram-v0 -> Breakout-ram-v0


NUM_RAND_INSTANCES = 5000

ENV_MAP = {
    "2d_mountain_car": MountainCarEnv(),
    "3d_mountain_car": ThreeDMountainCarEnv(),
    "2d_cart_pole":CartPoleEnv(),
    "3d_cart_pole": ThreeDCartPoleEnv(),
    "acrobot": AcrobotEnv()
}

SOURCE_MAP = {
    "2d_mountain_car": MountainCarEnv(),
    "acrobot": AcrobotEnv()
}
def get_rand_instance(env):
    env.reset()
    state = env.observation_space.sample()
    env.set_state(state)
    action = env.action_space.sample()
    print(state, action)
    next_state, reward, done, info = env.step(action)
    return [state, action, next_state, reward, done]


def generate_optimal_samples(env, modelpath):
    act = deepq.load(modelpath)
    print(act._act_params)
    q_values_graph = deepq.build_graph.build_q_values(**act._act_params)

    replay_memory = []  # reset
    for ep in range(100):  # 100 episodes
        obs, done = env.reset(), False
        result = []
        while not done:
            action = act(obs[None])[0]
            q_values = q_values_graph(obs[None])
            n_obs, rew, done, _ = env.step(action)
            print([obs, action, n_obs, rew, done, q_values[0][action]])
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
    for name, env in ENV_MAP.items():
        result = []
        print("Generating random samples  " + env.name)
        try:
            file_path = "../data/" + name + "/random_instances.pkl"
            if os.path.exists(file_path): continue
            for _ in range(NUM_RAND_INSTANCES):
                result.append(get_rand_instance(env))
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
            if os.path.exists(file_path): continue
            print("Generating optimal samples  " + env.name)
            with open(file_path, "wb+") as f:
                cPickcle.dump(generate_optimal_samples(env, "../models/" + name + "_dqn.pkl"), f)
        except Exception as e:
            print(e)
            traceback.print_exc()
    print("Done!")
    print("==============================")

def main():
    train_source_tasks()

    process_random_ins()
    process_optimal_ins()




def test_generated():


    names = ["2d_mountain_car", "3d_mountain_car", "2d_cart_pole", "3d_cart_pole", "acrobot"]
    # Read example
    # Data format:
    # Arrays of instances. Each instance:
    # [state, action, next_state, reward, done]
    # Print random instances:
    for name in names:
        with open("../data/" + name + "/random_instances.pkl", "rb") as f:
            result = cPickcle.load(f)
        print("------  " + name + "   -------")
        print(result[:3])
        print("------------------------------------")

    # Read example optimals samples
    # Data format:
    # Arrays of episodes data. Each episode consists all instances.

    # Print optimals
    names = ["2d_mountain_car", "acrobot"]
    for name in names:
        with open("../data/" + name + "/optimal_instances.pkl") as f:
            episodes = cPickcle.load(f)
            lengths = [len(episode) for episode in episodes]
        print("------  " + name + "   -------")
        print(np.asarray(lengths).sum())
        print(episodes[-1][:10])
        print("------------------------------------")

if __name__ == "__main__":
    main()
    test_generated()

