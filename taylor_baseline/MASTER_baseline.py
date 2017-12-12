from __future__ import print_function
import gym
import itertools
import matplotlib
import numpy as np
import tensorflow as tf
from lib.env.threedmountain_car import ThreeDMountainCarEnv
import lib.RandomAction
from lib.env.mountain_car import MountainCarEnv
import matplotlib.pyplot as plt
import os
import lib.qlearning as ql
import pickle
import deepq
from lib.env.cartpole import CartPoleEnv
from lib.env.threedcartpole import ThreeDCartPoleEnv

# Create model
def neural_net(x, weights, biases):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


def one_step_transition_model(learning_rate=0.1, n_hidden_1 = 32, n_hidden_2 = 32, num_input = 5, num_output = 4):

    # tf Graph input
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_output])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, num_output]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_output]))
    }

    # Construct model
    logits = neural_net(X, weights, biases)

    # Define loss and optimizer
    loss_op = tf.losses.mean_squared_error(logits, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    return loss_op, train_op, X, Y


def get_train_test_data(source_qlearn=True, source_env=MountainCarEnv(), target_env=ThreeDMountainCarEnv()):

    # source task
    if source_qlearn: # collect data from qlearning = true, collect data from random actions = false
        source_filename = './' + source_env.name + '_dsource_qlearn.npz'
        if os.path.isfile(source_filename):
            f_read = np.load(source_filename)
            dsource = f_read['dsource']

        else:
            model = deepq.models.mlp([64], layer_norm=True)
            act = deepq.learn(
                source_env,
                q_func=model,
                lr=1e-3,
                max_timesteps=40000,
                buffer_size=50000,
                exploration_fraction=0.1,
                exploration_final_eps=0.1,
                print_freq=1,
                param_noise=False
            )

            replay_memory = []  # reset
            for ep in range(100): # 100 episodes
                obs, done = source_env.reset(), False
                while not done:
                    n_obs, rew, done, _ = source_env.step(act(obs[None])[0])
                    replay_memory.append([obs, act(obs[None])[0], n_obs, rew, done])
                    obs = n_obs

            dsource = np.array(replay_memory)
            np.savez(source_filename, dsource=dsource)
            # with open('./data/q_learning.pkl', 'wb') as file:
            #     pickle.dump(qlearning_2d, file)
    else:
        source_filename = './' + source_env.name + '_dsource_random.npz'
        if os.path.isfile(source_filename):
            f_read = np.load(source_filename)
            dsource = f_read['dsource']
        else:
            qlearning_2d = lib.RandomAction.RandomAction(source_env)
            dsource = np.array(qlearning_2d.play())
            np.savez(source_filename, dsource=dsource)

    # target task
    target_filename = './' + target_env.name + '_dtarget_random.npz'
    if os.path.isfile(target_filename):
        f_read = np.load(target_filename)
        # print(f_read['dtarget'].shape)
        dtarget = f_read['dtarget']
    else:
        random_action_3d = lib.RandomAction.RandomAction(target_env)
        dtarget = np.array(random_action_3d.play())
        np.savez(target_filename, dtarget=dtarget)

    # Define the input function for training
    dsa = np.array([np.append(x[0], x[1]) for x in dtarget]) # dsa = d states actions
    dns = np.array([x[2] for x in dtarget]) # dns = d next states

    dsa_train = dsa[:-100]
    dns_train = dns[:-100]
    dsa_test = dsa[-100:]
    dns_test = dns[-100:]

    return dsa_train, dns_train, dsa_test, dns_test, dsource, dtarget


def train_model(num_steps=10000, batch_size=100, display_step=100, source_env=MountainCarEnv(),
                target_env=ThreeDMountainCarEnv()):
    loss_op, train_op, X, Y = one_step_transition_model(num_input=target_env.observation_space.shape[0]+1, num_output=target_env.observation_space.shape[0])
    dsa_train, dns_train, dsa_test, dns_test, dsource, dtarget = get_train_test_data(
        source_qlearn=False, source_env=source_env, target_env=target_env)

    batch_num = np.size(dsa_train, 0)

    init = tf.global_variables_initializer()
    loss = []

    saver = tf.train.Saver()

    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        for step in range(num_steps):
            batch_x = dsa_train[(step * batch_size) % batch_num: (step * batch_size + batch_size) % batch_num, :]
            batch_y = dns_train[(step * batch_size) % batch_num: (step * batch_size + batch_size) % batch_num, :]

            # Run optimization op (backprop)
            loss_train, _ = sess.run([loss_op, train_op], feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0:
                print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss_train))
                loss.append(loss_train)

        print("Optimization Finished!")

        # test set
        loss_test = sess.run(loss_op, feed_dict={X: dsa_test, Y: dns_test})
        print("test loss is {}".format(loss_test))

        save_path = saver.save(sess, "./data/tmp/model.ckpt")
        print("Model saved in file: %s" % save_path)

        # Find the mapping between source and target
        source_states = source_env.observation_space.shape[0]  # 2
        target_states = target_env.observation_space.shape[0]  # 4
        source_actions = source_env.action_space.n  # 3
        target_actions = target_env.action_space.n  # 5

        mse_state_mappings = np.zeros((source_states,) * target_states)  # 2 by 2 by 2 by 2
        mse_action_mappings = np.ndarray(shape=(target_actions, source_actions, pow(target_states, source_states)))  # 5 by 3 by 16
        mse_action_mappings.fill(-1)

        state_count = 0


        for target_states_list in itertools.product(range(source_states), repeat=target_states):
            state_losses = []
            for t_action in range(target_actions):
                for s_action in range(source_actions):
                    states = np.array([x[0] for x in dsource if x[1] == s_action])
                    actions = np.array([x[1] for x in dsource if x[1] == s_action])
                    n_states = np.array([x[2] for x in dsource if x[1] == s_action])

                    if (states.size == 0) or (actions.size == 0) or (n_states.size == 0):
                        print(
                            'this happened.. dsource does not have certain states or does not perform certain actions at all. make sure to generate better samples. possibly with high epsilon value')
                        # mse_action_mappings[t_action, s_action, state_count] = 0
                        continue

                    # transform to dsource_trans
                    actions_trans = np.ndarray(shape=(actions.size,))
                    actions_trans.fill(t_action)
                    input_trans = np.concatenate((states[:, target_states_list], actions_trans[:,None]), axis=1)
                    n_states_trans = np.squeeze(np.array([n_states[:, target_states_list]]))

                    # calculate mapping error
                    loss_mapping = sess.run(loss_op, feed_dict={X: input_trans, Y: n_states_trans})
                    # print('loss_mapping is {}'.format(loss_mapping))

                    state_losses.append(loss_mapping)
                    # import pdb; pdb.set_trace()
                    mse_action_mappings[t_action, s_action, state_count] = loss_mapping

            # import pdb; pdb.set_trace()
            mse_state_mappings[target_states_list] = np.mean(state_losses)
            state_count += 1

        ## mse_action_mappings_result = [[np.mean(mse_action_mappings[t_action, s_action, :]) for s_action in range(source_actions)] for t_action in range(target_actions)]

        mse_action_mappings_result = np.zeros((target_actions, source_actions))
        for t_action in range(target_actions):
            for s_action in range(source_actions):
                losses_act = []
                for s in range(target_states * target_states):
                    if mse_action_mappings[t_action, s_action, s] != -1:
                        # print(mse_action_mappings[t_action, s_action, s])
                        losses_act.append(mse_action_mappings[t_action, s_action, s])
                mse_action_mappings_result[t_action, s_action] = np.mean(losses_act)

        print('action mapping: {}'.format(mse_action_mappings_result))
        print('state mapping {}'.format(mse_state_mappings))

        count = 0
        for target_states_list in itertools.product(range(source_states), repeat=target_states):
            print(str(count) + ': ')
            print(mse_state_mappings[target_states_list])
            count += 1

        with open('./data/mse_state_mappings_3d_2d.pkl', 'wb') as file:
            pickle.dump(mse_state_mappings, file)

        with open('./data/mse_action_mappings_3d_2d.pkl', 'wb') as file:
            pickle.dump(mse_action_mappings, file)

        print("Done exporting MSE file")


if __name__ == '__main__':
    # train_model(num_steps=10000, batch_size=100, display_step=100, source_env=MountainCarEnv(),
    #             target_env=ThreeDMountainCarEnv())
    train_model(num_steps=10000, batch_size=100, display_step=100, source_env=ThreeDMountainCarEnv(),
                target_env=MountainCarEnv())
