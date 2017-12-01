import pickle
from matplotlib import pyplot as plt
import numpy as np


def plot_rewards_steps(with_transfer=True):

    name_str = 'with_transfer' if with_transfer else 'no_transfer'



if __name__ == '__main__':

    with open('exp_data/mountain_car_no_transfer.pkl', 'rb') as f:
        rewards, steps = pickle.load(f)

    rewards = np.asarray(rewards)
    steps = np.asarray(steps)
    plt.figure()
    plt.errorbar(np.arange(len(rewards[0])) + 1, rewards.mean(axis=0), yerr=rewards.std(axis=0), barsabove=True)

    plt.show()

