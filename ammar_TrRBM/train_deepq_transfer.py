import pickle
from TrRBM_train_models import train_dqn
import tensorflow as tf
from envs import ENVS_PATH_DICTIONARY


def train_dqn_transfer(source_env_str, target_env_str, num_episodes=100, num_experiments=5, with_transfer=True,
                       option_str='random'):

    with open('exp_data/{}_{}_{}_Transferred_Optimal.pkl'.format(source_env_str, target_env_str, option_str), 'rb') as f:
        target_states, target_actions, rewards, target_states_prime = pickle.load(f)

    name_str = 'with_transfer' + '_' + option_str if with_transfer else 'no_transfer'

    rewards_list, steps_list = [], []
    target_env = ENVS_PATH_DICTIONARY[target_env_str]['env']()
    for i in range(num_experiments):
        with tf.variable_scope("{}_num_{}".format(name_str, i)):
            rews, steps = train_dqn(target_env, target_states, target_actions, rewards, target_states_prime,
                                    with_transfer=with_transfer, max_episodes=num_episodes)
        rewards_list.append(rews)
        steps_list.append(steps)

    with open('exp_data/{}_{}_{}.pkl'.format(source_env_str, target_env_str, name_str), 'wb') as f:
        pickle.dump([rewards_list, steps_list], f)
    print('saved results')


if __name__ == '__main__':
    # train_dqn_transfer('2DMountainCar', '3DMountainCar', num_episodes=100, num_experiments=10, with_transfer=False,
    #                    option_str='realistic')  # option_str doesn't matter in this case
    # train_dqn_transfer('2DMountainCar', '3DMountainCar', num_episodes=100, num_experiments=10, with_transfer=True,
    #                    option_str='realistic')
    train_dqn_transfer('2DMountainCar', '3DMountainCar', num_episodes=100, num_experiments=10, with_transfer=True,
                       option_str='random')
    # train_dqn_transfer('2DMountainCar', '2DCartPole', num_episodes=100, num_experiments=10, with_transfer=True)
