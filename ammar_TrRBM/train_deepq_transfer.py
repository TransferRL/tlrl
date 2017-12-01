import pickle
from TrRBM_train_models import train_dqn
import tensorflow as tf


def train_dqn_transfer(source_env_str, target_env_str, num_episodes=100, num_experiments=5, with_transfer=True):

    with open('exp_data/{}_{}_Transferred_Optimal.pkl'.format(source_env_str, target_env_str), 'rb') as f:
        target_states, target_actions, rewards, target_states_prime = pickle.load(f)

    rewards_list, steps_list = [], []
    for i in range(num_experiments):
        with tf.variable_scope("num_{}".format(i)):
            rews, steps = train_dqn(target_states, target_actions, rewards, target_states_prime,
                                    with_transfer=with_transfer, max_episodes=num_episodes)
        rewards_list.append(rews)
        steps_list.append(steps)

    name_str = 'with_transfer' if with_transfer else 'no_transfer'

    with open('exp_data/mountain_car_{}.pkl'.format(name_str), 'wb') as f:
        pickle.dump([rewards_list, steps_list], f)


if __name__ == '__main__':
    train_dqn_transfer(num_episodes=10, num_experiments=3, with_transfer=True)
