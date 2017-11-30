import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from matplotlib import pyplot as plt

import deepq_mod
import baselines.deepq as deepq
import trrbm
import pickle

from envs import ENVS_DICTIONARY

N_MAPPED = 5000
target_env = ENVS_DICTIONARY['3DMountainCar']()
_3d = True

params_dictionary = {}
params_dictionary["discount_rate"] = 0.9
params_dictionary["mem_size"] = 400
params_dictionary["sample_size"] = 200
params_dictionary["n_hidden_layers"] = 2
params_dictionary["n_hidden_units"] = 16
params_dictionary["activation"] = tf.nn.relu
params_dictionary["optimizer"] = tf.train.MomentumOptimizer
params_dictionary["opt_kws"] = {'learning_rate': 0.001, 'momentum': 0.2}
params_dictionary["n_episodes"] = 100
params_dictionary["n_epochs"] = 25
params_dictionary["retrain_period"] = 1
params_dictionary["epsilon"] = 0.5
params_dictionary["epsilon_decay"] = 0.999
params_dictionary["ini_steps_retrain"] = 50

params_dictionary["TrRBM_hidden_units"] = 150
params_dictionary["TrRBM_batch_size"] = 100
params_dictionary["TrRBM_learning_rate"] = 0.000001
params_dictionary["TrRBM_num_epochs"] = 20
params_dictionary["TrRBM_n_factors"] = 100
params_dictionary["TrRBM_k"] = 1
params_dictionary["TrRBM_use_tqdm"] = True
params_dictionary["TrRBM_show_err_plt"] = False

render = False


def load_samples(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def unpack_samples(samples, action_encoder, fit_encoder=True):
    """
    to take source and target random samples output by taylor_master/lib/instance_sampler.py
    and format in model-ready vectors
    """
    unpacked = []
    actions = []
    for sample in samples:
        state = np.array(sample[0])
        state_prime = np.array(sample[2])
        reward = np.array(sample[3])
        terminal = np.array(int(sample[4]))

        # need to one-hot-encode the action
        action = [int(sample[1])]

        if action_encoder is not None:
            unpacked.append(np.concatenate([state, state_prime]))
            actions.append(action)
        else:
            unpacked.append(np.concatenate([state, state_prime, action]))

    if action_encoder is not None:
        if fit_encoder:
            action_encoder.fit(np.array(actions).reshape(-1, 1))
        actions = action_encoder.transform(np.array(actions).reshape(-1, 1)).astype(float)
        unpacked = np.concatenate([unpacked, actions], axis=1)
    else:
        unpacked = np.stack(unpacked)
    return action_encoder, unpacked


def unpack_episodes(episodes, action_encoder, fit_encoder=False):
    """
    to take source optimal samples (list of episodes) and format in model-ready vectors
    """
    samples = []
    for episode in episodes:
        for sample in episode:
            samples.append(sample)
    _, unpacked = unpack_samples(samples, action_encoder, fit_encoder=fit_encoder)
    return unpacked


def even_out_samplesizes(samples1, samples2):
    """
    if there are more of source/target task samples than the other - randomly tiles the lesser
    sample size to even out sample sizes
    """
    if len(samples1) < len(samples2):
        samples1 = np.tile(samples1, [np.math.ceil(len(samples2) / (len(samples1))), 1])
        np.random.shuffle(samples1)
        samples1 = samples1[:len(samples2)]

    if len(samples1) > len(samples2):
        samples2 = np.tile(samples2, [np.math.ceil(len(samples1) / (len(samples2))), 1])
        np.random.shuffle(samples2)
        samples2 = samples2[:len(samples1)]

    return samples1, samples2


def prepare_target_triplets(target_mapped, state_size, action_size):
    """
    to separate state, action, and transition state matrixes
    """

    target_states = target_mapped[:, :state_size]
    target_states_prime = target_mapped[:, state_size:-action_size]
    target_actions = target_mapped[:, -action_size:]
    actions_greedy = np.argmax(target_actions, axis=1)

    # actions_probabilities = (np.exp(-target_actions)/np.sum(np.exp(-target_actions),axis=1).reshape(-1,1))
    # actions_sampled = np.random.multinomial(1, actions_probabilities[0], N_ACT_SAMPLES).argmax(1)

    return target_states, target_states_prime, actions_greedy.reshape(-1, 1)


def generate_rewards(env, states, actions):
    """
    runs step in actual environment to generate reward for that (s,a) tuple
    COMMENT: this is completely unrealistic and also inaccurate; inaccurate since the reward
    should be associated with the (s,a,s') tuple, and unrealistic b/c in real life it would (most likley)
    be impossible to chose the state an agent is in outside bounds of moving through state space
    one step at a time
    """
    # for now just taking the actual environment as the black box function
    assert len(states) == len(actions)
    rewards = []
    for state, action in zip(states, actions):
        env.state = state
        next_state, reward, done, info = env.step(action[0])
        rewards.append(reward)
    return np.array(rewards).reshape(-1, 1)


def plot_with_gp(X, y, title='', xlabel='', ylabel='', plt_lable='', color='b'):
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(X, y)
    X_ = np.linspace(min(X), max(X), 100)
    y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)

    plt.plot(X_, y_mean, color=color, lw=3, zorder=9, label=plt_lable)
    plt.fill_between(X_, y_mean - y_std, y_mean + y_std,
                     alpha=0.2, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.show()


def train_dqn(target_states, target_actions, rewards, target_states_prime, with_transfer=True, max_episodes=100):
    # run multiple experiments with the same transfer instances
    model = deepq.models.mlp([64], layer_norm=True)

    dq = deepq_mod.DeepQ(
        target_env,
        q_func=model,
        lr=1e-3,
        max_timesteps=50000,
        buffer_size=25000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        print_freq=10,
        param_noise=False, max_episodes=max_episodes)

    # use transferred tuples to learn initial target policy \pi_{T}^{o}

    dq.make_build_train()

    dq.initialize()
    if with_transfer:
        dq.transfer_pretrain(
            zip(list(target_states), list(target_actions.squeeze()), list(rewards.squeeze()),
                list(target_states_prime))
            , epochs=100
            , tr_batch_size=32
            , keep_in_replay_buffer=False
        )

    # use initial target policy and learn as we go
    act, episode_rewards, episode_steps = dq.task_train()
    return episode_rewards, episode_steps


def main():
    source_random_path = '../taylor_baseline/data/2d_instances.pkl'
    target_random_path = '../taylor_baseline/data/3d_instances.pkl'
    source_optimal_path = '../taylor_baseline/data/optimal_instances.pkl'

    # load source task random samples
    source_action_encoder, source_random = unpack_samples(load_samples(source_random_path), OneHotEncoder(sparse=False))

    # load target task random samples
    target_action_encoder, target_random = unpack_samples(load_samples(target_random_path), OneHotEncoder(sparse=False))

    # prepare samples
    source_random, target_random = even_out_samplesizes(source_random, target_random)
    # source_scaler, source_random = utils.standardize_samples(source_random)
    # target_scaler, target_random = utils.standardize_samples(target_random)

    # load the TrRBM model

    rbm = trrbm.RBM(
        name="TrRBM",
        v1_size=source_random.shape[1],
        h_size=params_dictionary["TrRBM_hidden_units"],
        v2_size=target_random.shape[1],
        n_data=source_random.shape[0],
        batch_size=params_dictionary["TrRBM_batch_size"],
        learning_rate=params_dictionary["TrRBM_learning_rate"],
        num_epochs=params_dictionary["TrRBM_num_epochs"],
        n_factors=params_dictionary["TrRBM_n_factors"],
        k=params_dictionary["TrRBM_k"],
        use_tqdm=params_dictionary["TrRBM_use_tqdm"],
        show_err_plt=params_dictionary["TrRBM_show_err_plt"]
    )

    # train the TrRBM model
    errs = rbm.train(source_random, target_random)

    if rbm.show_err_plt:
        plt.plot(range(len(rbm.cost)), rbm.cost)
        plt.title('TrRBM training reconstruction error')
        plt.xlabel('epoch')
        plt.ylabel('avg reconstruction error')
        plt.show()

    # load source task optimal instances
    source_optimal = unpack_episodes(load_samples(source_optimal_path), source_action_encoder, fit_encoder=False)
    # source_optimal = source_scaler.transform(source_optimal)

    # map to target instances
    print('DEBUG: mapping instances over using TrRBM')
    np.random.shuffle(source_optimal)
    target_mapped = rbm.v2_predict(source_optimal[:N_MAPPED])
    # target_mapped = target_scaler.inverse_transform(target_mapped)

    # prepare target instances (i.e. decode action; split s from s')
    print('DEBUG: preparing target instances')
    action_size = int(target_action_encoder.feature_indices_[-1])
    state_size = int((target_mapped.shape[1] - target_action_encoder.feature_indices_[-1]) / 2)
    target_states, target_states_prime, target_actions = prepare_target_triplets(target_mapped, state_size, action_size)

    # get rewards from black-box model of reward function
    print('DEBUG: generating black-box rewards')
    rewards = generate_rewards(target_env, target_states, target_actions)

    with open('exp_data/3DmountainCarTarget.pkl', 'wb') as f:
        pickle.dump([target_states, target_actions, rewards, target_states_prime], f)

    print('saved TrRBM outputs')
    # TODO: one alternative to getting rewards from black-box model may be using (normalized?) Q values from source task

    # build target policy Q value function approximator
    #
    # m_rewards, steps = train_dqn(target_states, target_actions, rewards, target_states_prime, with_transfer=True)
    #
    # with open('exp_data/mountainCarTransfer.pkl', 'wb') as f:
    #     pickle.dump([m_rewards, steps], f)


    # output results for persistance
    # TODO


if __name__ == '__main__':
    main()
    print('done')
