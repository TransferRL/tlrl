import pickle
from matplotlib import pyplot as plt
from TrRBM_2d_3d_rbmtrain import plot_with_gp
import numpy as np

if __name__ == '__main__':

    with open('exp_data/mountain_car_no_transfer.pkl', 'rb') as f:
        rewards, steps = pickle.load(f)

    rewards = np.asarray(rewards)
    steps = np.asarray(steps)
    plt.figure()
    plt.errorbar(np.arange(len(rewards[0])) + 1, rewards.mean(axis=0), yerr=rewards.std(axis=0), barsabove=True)

    plt.show()

