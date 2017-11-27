import pickle
from TrRBM_2d_3d_plots import train_dqn


def main():
    with open('exp_data/3DmountainCarTarget.pkl', 'rb') as f:
        target_states, target_actions, rewards, target_states_prime = pickle.load(f)

    X, m_rewards, steps = train_dqn(target_states, target_actions, rewards, target_states_prime, with_transfer=False,
                                    num_experiments=5)
    with open('exp_data/mountainCarNoTransfer.pkl', 'wb') as f:
        pickle.dump([X, m_rewards, steps], f)


if __name__ == '__main__':
    main()
