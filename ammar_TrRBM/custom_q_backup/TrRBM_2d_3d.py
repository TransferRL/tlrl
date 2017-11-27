import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
import gym
from tqdm import tqdm
import collections

import sys
sys.path.append('../taylor_master/lib/env/')
from threedmountain_car import ThreeDMountainCarEnv

from dqn import q_network
import trrbm
import utils

import pickle
import datetime

N_MAPPED = 5000
target_env = ThreeDMountainCarEnv()
_3d = True

params_dictionary = {}
params_dictionary["discount_rate"] = 0.9
params_dictionary["mem_size"] = 400
params_dictionary["sample_size"] = 200
params_dictionary["n_hidden_layers"] = 2
params_dictionary["n_hidden_units"] = 16
params_dictionary["activation"] = tf.nn.relu
params_dictionary["optimizer"] = tf.train.MomentumOptimizer
params_dictionary["opt_kws"] = {'learning_rate':0.001,'momentum':0.2}
params_dictionary["n_episodes"] = 100
params_dictionary["n_epochs"] = 25
params_dictionary["retrain_period"] = 1
params_dictionary["epsilon"] = 0.5
params_dictionary["epsilon_decay"] = 0.999
params_dictionary["ini_steps_retrain"] = 50

params_dictionary["TrRBM_hidden_units"] = 100
params_dictionary["TrRBM_batch_size"] = 100
params_dictionary["TrRBM_learning_rate"] = 0.000001
params_dictionary["TrRBM_num_epochs"] = 100
params_dictionary["TrRBM_n_factors"] = 40
params_dictionary["TrRBM_k"] = 1
params_dictionary["TrRBM_use_tqdm"] = True
params_dictionary["TrRBM_show_err_plt"] = True

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
            unpacked.append(np.concatenate([state,state_prime]))
            actions.append(action)
        else:
            unpacked.append(np.concatenate([state,state_prime,action]))
            
    if action_encoder is not None:
        if fit_encoder == True:
            action_encoder.fit(np.array(actions).reshape(-1,1))
        actions = action_encoder.transform(np.array(actions).reshape(-1,1)).astype(float)
        unpacked = np.concatenate([unpacked,actions],axis=1)
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
    _, unpacked = unpack_samples(samples, action_encoder, fit_encoder=False)
    return unpacked
    
def even_out_samplesizes(samples1, samples2):
    """
    if there are more of source/target task samples than the other - randomly tiles the lesser
    sample size to even out sample sizes
    """
    if len(samples1) < len(samples2):
        samples1 = np.tile(samples1,[np.math.ceil(len(samples2)/(len(samples1))),1])
        np.random.shuffle(samples1)
        samples1 = samples1[:len(samples2)]
        
    if len(samples1) > len(samples2):
        samples2 = np.tile(samples2,[np.math.ceil(len(samples1)/(len(samples2))),1])
        np.random.shuffle(samples2)
        samples2 = samples2[:len(samples1)]
        
    return samples1, samples2


def prepare_target_triplets(target_mapped,state_size,action_size):
    """
    to separate state, action, and transition state matrixes
    """

    target_states = target_mapped[:,:state_size]
    target_states_prime = target_mapped[:,state_size:-action_size]
    target_actions = target_mapped[:,-action_size:]
    actions_greedy = np.argmax(target_actions,axis=1)

    #actions_probabilities = (np.exp(-target_actions)/np.sum(np.exp(-target_actions),axis=1).reshape(-1,1))
    #actions_sampled = np.random.multinomial(1, actions_probabilities[0], N_ACT_SAMPLES).argmax(1)

    return target_states, target_states_prime, actions_greedy.reshape(-1,1)

def generate_rewards(env,states,actions):
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
    for state, action in zip(states,actions):
        env.state = state
        next_state, reward, done, info = env.step(action[0])
        rewards.append(reward)
    return np.array(rewards).reshape(-1,1)
    

    
def main():
    
    source_random_path = '../taylor_master/data/2d_instances.pkl'
    target_random_path = '../taylor_master/data/3d_instances.pkl'
    source_optimal_path = '../taylor_master/data/optimal_instances.pkl'

    # load source task random samples
    source_action_encoder, source_random = unpack_samples(load_samples(source_random_path),OneHotEncoder(sparse=False))

    # load target task random samples
    target_action_encoder, target_random = unpack_samples(load_samples(target_random_path),OneHotEncoder(sparse=False))


    # prepare samples
    source_random, target_random = even_out_samplesizes(source_random, target_random)
    source_scaler, source_random = utils.standardize_samples(source_random)
    target_scaler, target_random = utils.standardize_samples(target_random)

    # load the TrRBM model

    rbm = trrbm.RBM(
        name = "TrRBM",
        v1_size = source_random.shape[1], 
        h_size = params_dictionary["TrRBM_hidden_units"], 
        v2_size = target_random.shape[1], 
        n_data = source_random.shape[0], 
        batch_size = params_dictionary["TrRBM_hidden_units"], 
        learning_rate = params_dictionary["TrRBM_learning_rate"],
        num_epochs = params_dictionary["TrRBM_num_epochs"], 
        n_factors = params_dictionary["TrRBM_n_factors"],
        k = params_dictionary["TrRBM_k"],
        use_tqdm = params_dictionary["TrRBM_use_tqdm"],
        show_err_plt = params_dictionary["TrRBM_show_err_plt"]
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
    source_optimal = source_scaler.transform(source_optimal)

    # map to target instances
    print('DEBUG: mapping instances over using TrRBM')
    np.random.shuffle(source_optimal)
    target_mapped = rbm.v2_predict(source_optimal[:N_MAPPED])
    target_mapped = target_scaler.inverse_transform(target_mapped)

    # prepare target instances (i.e. decode action; split s from s')
    print('DEBUG: preparing target instances')
    action_size = int(target_action_encoder.feature_indices_[-1])
    state_size = int((target_mapped.shape[1]-target_action_encoder.feature_indices_[-1])/2)
    target_states, target_states_prime, target_actions = prepare_target_triplets(target_mapped,state_size,action_size)

    # get rewards from black-box model of reward function
    print('DEBUG: generating black-box rewards')
    rewards = generate_rewards(target_env,target_states,target_actions)
    
    # TODO: one alternative to getting rewards from black-box model may be using (normalized?) Q values from source task

    # use transferred tuples to learn initial target policy \pi_{T}^{o} (as Q network)

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

    dqn.add_new_obvs(target_states, target_actions, target_states_prime, rewards)
    _states, _actions, _transitions, _rewards = dqn.get_memory_sample(dqn.mem_size)
    dqn.run_training(150, _states, _actions, _transitions, _rewards)
    dqn.plot_loss() 


    # use initial target policy and learn as we go

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
        state = np.array(target_env.reset()).reshape(1,-1)
        done = False
        while True:
            steps_counter['steps'] += 1
            episode_counter[episode] += 1
            # epsilon-greedily take next action from network
            if np.random.random_sample() > EPSILON:
                action = dqn.get_next_action(state)[0]
            else:
                action = target_env.action_space.sample()
            next_state, reward, done, _ = target_env.step(action)
            
            state = np.array(next_state).reshape(1,-1)
            
            print('episode:', episode, 'steps:', episode_counter[episode], 'state:', next_state)
            
            episode_total_reward[episode] += reward
            rewards.append(reward)
            
            if _3d == True and RENDER == True:
                env.render_orthographic()
            elif RENDER == True:
                env.render()
            dqn.add_new_obvs(state.reshape(1,-1),np.array([action]).reshape(1,-1),next_state.reshape(1,-1),np.array(reward).reshape(1,-1))
            if steps_counter['steps'] == INI_STEPS_RETRAIN or (steps_counter['steps'] > INI_STEPS_RETRAIN and steps_counter['steps'] % RETRAIN_PERIOD == 0):
                print(steps_counter['steps'])
                _states, _actions, _transitions, _rewards = dqn.get_memory_sample(dqn.mem_size)
                dqn.run_training(N_EPOCHS, _states, _actions, _transitions, _rewards)
            #instances.append([state,action,next_state,reward,done])
            
            if episode > 1:
                EPSILON = EPSILON_DECAY*EPSILON
                
            if done == True:
                print('episode {} completed'.format(len(episode_counter)))
                break

        if len(episode_counter) > 20 and np.all(np.array(list(episode_counter)[-20:]) <= 1000) == True:
            break
            
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

    file = 'exp_data/TrRBM_exp_{}.p'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    with open(file, 'wb') as f:
        pickle.dump(output,f)
    
    print('DONE!')
    
if __name__ == '__main__':
    instances, episode_counter =  main()
    print('done')