# Not formal tests
import sys
import os
import pickle as cPickcle
import lib.env.mountain_car
import lib.qlearning as ql
import numpy as np
import tensorflow as tf
import baselines.common.tf_util as U
import deepq

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


from lib.instance_sampler import InstanceSampler

def generate_2D_sampler(num):
    s = InstanceSampler()
    result = []
    for i in range(num):
        result.append(s.getRandom2DInstance())
    return result


def generate_3D_sampler(num):
    s = InstanceSampler()
    result = []
    for i in range(num):
        result.append(s.getRandom3DInstance())
    return result

def generate_2D_optimal_samplers(num):
    result = []
    env = lib.env.mountain_car.MountainCarEnv()
    model = deepq.models.mlp([64], layer_norm=True)
    # act = deepq.learn(
    #     env,
    #     q_func=model,
    #     lr=1e-3,
    #     max_timesteps=40000,
    #     buffer_size=50000,
    #     exploration_fraction=0.1,
    #     exploration_final_eps=0.1,
    #     print_freq=1,
    #     param_noise=False
    # )

    # act.save("2D_Mountain_Car_dqn_model.pkl")

    act = deepq.load("2D_Mountain_Car_dqn_model.pkl")
    print(act._act_params)
    q_values_graph = deepq.build_graph.build_q_values(**act._act_params)
    # sess = tf.Session()
    # sess.run(q_values_graph)
    # print(q_values_graph)
    # # # print(act._act)
    # exit()

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

def get_q_values(make_obs_ph, q_func, num_actions, scope="q-values", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        q_values = q_func(observations_ph.get(), num_actions, scope="q_func")
        return q_values

if __name__ == "__main__":
    with open('./data/optimal_instances_with_q_value.pkl', 'wb+') as f:
        cPickcle.dump(generate_2D_optimal_samplers(100), f);
    # with open('./data/2d_instances.pkl', "wb+") as f:
    #     cPickcle.dump(generate_2D_sampler(5000), f)
    # with open('./data/3d_instances.pkl', "wb+") as f:
    #     cPickcle.dump(generate_3D_sampler(5000), f)

    # Read example
    # Data format:
    # Arrays of instances. Each instance:
    # [state, action, next_state, reward, done]
    # with open('./data/2d_instances.pkl', "rb") as f:
    #     instances = cPickcle.load(f)
    #     count_terminations = [1 for ele in instances if ele[4]]
    #     print(len(count_terminations))
    # with open('./data/3d_instances.pkl', "rb") as f:
    #     instances = cPickcle.load(f)
    #     count_terminations = [1 for ele in instances if ele[4]]
    #     print(len(count_terminations))


    # Read example optimals samples
    # Data format:
    # Arrays of episodes data. Each episode consists all instances.
    with open('./data/optimal_instances_with_q_value.pkl', "rb") as f:
        episodes = cPickcle.load(f)
        lengths = [len(episode) for episode in episodes]
        print(np.asarray(lengths).sum())