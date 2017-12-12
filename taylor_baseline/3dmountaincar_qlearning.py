import itertools
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from lib.env.threedmountain_car import ThreeDMountainCarEnv
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
import sklearn.pipeline
from lib import plotting
import pickle
import pprint

if "./" not in sys.path:
    sys.path.append("./")

# Load State MSE Mapping objects
with open('data/mse_state_mappings.pkl', 'rb') as file:
    mse_state_mappings = pickle.load(file)

with open('data/mse_state_mappings_3d_2d.pkl', 'rb') as file:
    mse_state_mappings_3d_2d = pickle.load(file)

with open('data/mse_action_mappings.pkl', 'rb') as file:
    mse_action_mappings = pickle.load(file)

with open('data/q_learning.pkl', 'rb') as file:
    qlearning_2d = pickle.load(file)
    historic_predictor = qlearning_2d.estimator.predict


pprint.pprint(mse_state_mappings)
pprint.pprint(mse_action_mappings)
pprint.pprint(historic_predictor)
print(np.shape(mse_state_mappings))
print(np.shape(mse_action_mappings))

TOP = 3
env = ThreeDMountainCarEnv()

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to convert a state to a featurizes representation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))


class Estimator():
    """
    Value Function approximator.
    """

    def __init__(self):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)

    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = scaler.transform([state])
        featurized = featurizer.transform(scaled)
        return featurized[0]

    def predict(self, s, a=None):
        """
        Makes value function predictions.

        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for

        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.

        """
        features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]

    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])


def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.5, epsilon_decay=1.0, transfer=True,
               render=True, theta=1.0, theta_discount_factor=1.0):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor
        transfer: Boolean to control transfer from 2d.
        render: Boolean to control rendering during training

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):

        # The policy we're following
        policy = make_epsilon_greedy_policy(
            estimator, epsilon * epsilon_decay ** i_episode, env.action_space.n)

        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = stats.episode_rewards[i_episode - 1]
        sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()

        # Only used for SARSA, not Q-Learning
        next_action = None

        theta = theta*theta_discount_factor
        # One step in the environment
        for t in itertools.count():

            # Choose an action to take
            # If we're using SARSA we already decided in the previous step
            if next_action is None:
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            else:
                action = next_action

            # Take a step
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD Update
            q_values_next = estimator.predict(next_state)

            # Get the best mapping for
            state_map_2d = np.zeros((TOP, len(mse_state_mappings)))
            top_indices = np.argpartition(np.ndarray.flatten(mse_state_mappings_3d_2d), TOP)[:TOP]
            for i, index in enumerate(top_indices):
                state_map_2d[i][0] = np.floor(index/len(mse_state_mappings_3d_2d))
                state_map_2d[i][1] = np.mod(index, len(mse_state_mappings_3d_2d))

            if transfer:
                q_values_historic = np.zeros(env.action_space.n)

                # For every possible action, learn historic q-values
                for current_action in range(env.action_space.n):
                    # Use historical environment from 2d Learning env
                    sum = 0
                    for source_action in range(len(mse_action_mappings[current_action])):
                        sum += np.mean(mse_action_mappings[current_action][source_action])

                    for source_action in range(len(mse_action_mappings[current_action])):
                        x, x_dot, y, y_dot = state
                        mse = np.mean(mse_action_mappings[current_action][source_action])

                        for map in state_map_2d:
                            mapped = [state[int(i)] for i in map]
                            q_values_historic[current_action] += historic_predictor(mapped)[source_action]*(1/sum)*(1/mse)

                        q_values_historic /= len(state_map_2d)

                # Historic update
                q_values_next += theta*q_values_historic

            # Use this code for Q-Learning
            # Q-Value TD Target
            td_target = reward + discount_factor * np.max(q_values_next)

            # Use this code for SARSA TD Target for on policy-training:
            # next_action_probs = policy(next_state)
            # next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            # td_target = reward + discount_factor * q_values_next[next_action]

            # Update the function approximator using our target
            estimator.update(state, action, td_target)

            print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, last_reward), end="")

            if render:
                env.render()
                env.render_y()

            if done:
                break

            if t > 100000:
                print("More than 100k steps")
                break

            state = next_state

    return stats


estimator = Estimator()

stats = q_learning(env, estimator, 100, epsilon=0.0, transfer=True, render=False, theta=0.5)
print(stats.episode_rewards)

estimator = Estimator()

stats2 = q_learning(env, estimator, 100, epsilon=0.0, transfer=False, render=False)
print(stats2.episode_rewards)

plotting.plot_two_episode_stats([stats, stats2], smoothing_window=25)
