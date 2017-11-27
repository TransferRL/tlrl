import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
import gym
from tqdm import tqdm
import collections

from dqn import q_network
import trrbm
import utils

import pickle
import datetime

"""
example of params_dictionary

params_dictionary = {}
params_dictionary["discount_rate"] = 0.9
params_dictionary["mem_size"] = 100
params_dictionary["sample_size"] = 50
params_dictionary["n_hidden_layers"] = 2
params_dictionary["n_hidden_units"] = 16
params_dictionary["activation"] = tf.nn.relu
params_dictionary["optimizer"] = tf.train.MomentumOptimizer
params_dictionary["opt_kws"] = {'learning_rate':0.01,'momentum':0.2}
params_dictionary["n_episodes"] = 500
params_dictionary["n_epochs"] = 5
params_dictionary["retrain_period"] = 1
params_dictionary["epsilon"] = 0.5
params_dictionary["epsilon_decay"] = 0.999
params_dictionary["ini_steps_retrain"] = 50
"""

def main(env, exp_name, state_size, action_size, params_dictionary, _3d=False,render=False):

    dqn = q_network(discount_rate = params_dictionary["discount_rate"]
                 ,mem_size = params_dictionary["mem_size"]
                 ,sample_size = params_dictionary["sample_size"]
                 ,n_input_units = state_size
                 ,n_output_units = action_size
                 ,n_hidden_layers = params_dictionary["n_hidden_layers"]
                 ,n_hidden_units = params_dictionary["n_hidden_units"]
                 ,activation = params_dictionary["activation"]
                 ,opt = params_dictionary["optimizer"]
                 ,opt_kws = params_dictionary["opt_kws"]
                 )

    dqn.initialize_graph()
    dqn.open_session()
    dqn.initialize_new_variables()
    
    N_EPISODES = params_dictionary["n_episodes"]
    N_EPOCHS = params_dictionary["n_epochs"]
    RETRAIN_PERIOD = params_dictionary["retrain_period"]
    EPSILON = params_dictionary["epsilon"]
    EPSILON_DECAY = params_dictionary["epsilon_decay"]
    INI_STEPS_RETRAIN = params_dictionary["ini_steps_retrain"]
    RENDER = render

    pbar = tqdm(range(N_EPISODES))
    episode_counter = collections.Counter()
    episode_total_reward = collections.Counter()
    steps_counter = collections.Counter()
    instances = []
    rewards = []
    for episode in pbar:
        state = np.array(env.reset()).reshape(1,-1)
        done = False
        while True:
            steps_counter['steps'] += 1
            episode_counter[episode] += 1
            # epsilon-greedily take next action from network
            if np.random.random_sample() > EPSILON:
                action = dqn.get_next_action(state)[0]
            else:
                action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            
            state = np.array(next_state).reshape(1,-1)
            
            print('episode:', episode, 'steps:', episode_counter[episode], 'state:', next_state)
            
            episode_total_reward[episode] += reward
            rewards.append(reward)
            
            if _3d == True and RENDER == True:
                env.render_orthographic()
            elif RENDER == True:
                env.render()
            dqn.add_new_obvs(state.reshape(1, -1), np.array([action]).reshape(1, -1), next_state.reshape(1, -1), np.array(reward).reshape(1, -1))
            if steps_counter['steps'] == INI_STEPS_RETRAIN or (steps_counter['steps'] > INI_STEPS_RETRAIN and steps_counter['steps'] % RETRAIN_PERIOD == 0):
                print('training qnet')
                _states, _actions, _transitions, _rewards = dqn.get_memory_sample(dqn.sample_size)
                print('gotten memory samples')
                dqn.run_training(N_EPOCHS, _states, _actions, _transitions, _rewards)
              
            EPSILON = EPSILON_DECAY*EPSILON

            if done == True:
                print('episode {} completed'.format(len(episode_counter)))
                break
                
            if episode_counter[episode] > 1000:
                if _3d == True:
                    env.render_orthographic()
                else:
                    env.render()
                
            
        if _3d == True and RENDER == True:
            #env.close_gui()
            pass
        elif RENDER == True:
            env.close()

    # plot results
    
    avg_rewards = np.array(list(episode_counter.values()))/np.array(list(episode_total_reward.values()))
    plt.scatter(list(episode_counter.keys()),avg_rewards)
    plt.title('avg reward over episodes')
    plt.xlabel('episode')
    plt.ylabel('mean reward')
    plt.show()
    
    plt.scatter(list(episode_counter.keys()),list(episode_counter.values()))
    plt.title('steps per episode')
    plt.xlabel('episode')
    plt.ylabel('steps')
    plt.show()
    

    # output results for persistance
    output = {'description':'this is a description of the experiment'
              ,'rewards':rewards
              ,'episode_counter':episode_counter
              ,'episode_total_reward':episode_total_reward
              ,'params':{}}

    file = 'exp_data/{}_exp_{}.p'.format(exp_name, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    with open(file, 'wb') as f:
        pickle.dump(output,f)
    
    print('DONE!')