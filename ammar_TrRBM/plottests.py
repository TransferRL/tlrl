import pickle
from matplotlib import pyplot as plt
import numpy as np


def plot_rewards_steps(source_env_str, target_env_str, with_transfer=True, option_str='random'):

    name_str = 'with_transfer' + '_' + option_str if with_transfer else 'no_transfer'
    title_str = '{} to {}'.format(source_env_str, target_env_str)

    with open('exp_data/{}_{}_{}.pkl'.format(source_env_str, target_env_str, name_str), 'rb') as f:
        rewards, steps = pickle.load(f)

    rewards = np.asarray(rewards)
    steps = np.asarray(steps)

    plt.figure(1)
    # plt.errorbar(np.arange(rewards.shape[1]) + 1, rewards.mean(axis=0), yerr=rewards.std(axis=0), label=name_str)
    plt.plot(np.arange(rewards.shape[1]) + 1, rewards.mean(axis=0), label=name_str)
    plt.title(title_str)

    # plt.figure(2)
    # plt.plot(np.arange(rewards.shape[1]) + 1, rewards.mean(axis=0), label=name_str)
    # plt.title(title_str)
    # plt.errorbar(np.arange(steps.shape[1]) + 1, steps.mean(axis=0), yerr=steps.std(axis=0), label='name_str')

def main():

    plot_rewards_steps('2DMountainCar', '3DMountainCar', with_transfer=True, option_str='random')
    plot_rewards_steps('2DMountainCar', '3DMountainCar', with_transfer=True, option_str='realistic')
    plot_rewards_steps('2DMountainCar', '3DMountainCar', with_transfer=False)
    plt.legend(loc=3)
    plt.show()

if __name__ == '__main__':

    main()
