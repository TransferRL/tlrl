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


# Parameters
learning_rate = 0.1
num_steps = 10000
batch_size = 100
display_step = 100

# Network Parameters
n_hidden_1 = 32 # 1st layer number of neurons
n_hidden_2 = 32 # 2nd layer number of neurons
num_input = 5
num_output = 4

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


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)

# Define loss and optimizer
loss_op = tf.losses.mean_squared_error(logits, Y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(logits), tf.round(Y)), tf.int32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()



# get envs
mc2d_env = MountainCarEnv()
mc3d_env = ThreeDMountainCarEnv()

# train source task
# qlearning_2d = ql.QLearning(mc2d_env)
# qlearning_2d.learn()
# dsource = qlearning_2d.play()

# 2d random
# random_action_2d = lib.RandomAction.RandomAction(mc2d_env)
# dsource = random_action_2d.play()


# print(dsource)

# do random action for target task
# random_action_3d = lib.RandomAction.RandomAction(mc3d_env)
# dtarget = random_action_3d.play()
# print(dtarget)


# source task
# if os.path.isfile('./dsource_qlearn.npz'):
#     f_read = np.load('./dsource_qlearn.npz')
#     # print(f_read['dsource'].shape)
#     dsource = f_read['dsource']
#
# else:
#     qlearning_2d = ql.QLearning(mc2d_env)
#     qlearning_2d.learn()
#     dsource = np.array(qlearning_2d.play())
#     # print(dsource.shape)
#     np.savez('dsource_qlearn.npz', dsource=dsource)
#
#     with open('./data/q_learning.pkl', 'wb') as file:
#         pickle.dump(qlearning_2d, file)

if os.path.isfile('./dsource_random.npz'):
    f_read = np.load('./dsource_random.npz')
    # print(f_read['dsource'].shape)
    dsource = f_read['dsource']
else:
    qlearning_2d = lib.RandomAction.RandomAction(mc2d_env)
    dsource = np.array(qlearning_2d.play())
    # print(dsource.shape)
    np.savez('dsource_random.npz', dsource=dsource)



# target task
if os.path.isfile('./dtarget_random.npz'):
    f_read = np.load('./dtarget_random.npz')
    # print(f_read['dtarget'].shape)
    dtarget = f_read['dtarget']
else:
    random_action_3d = lib.RandomAction.RandomAction(mc3d_env)
    dtarget = np.array(random_action_3d.play())
    np.savez('./dtarget_random.npz', dtarget=dtarget)



# approximate the one-step transition model
# Define the input function for training
dsa = np.array([np.append(x[0], x[1]) for x in dtarget])
dns = np.array([x[2] for x in dtarget])

# print(dsa.shape)
# print(dsa)

dsa_train = dsa[:-100]
dns_train = dns[:-100]
dsa_test = dsa[-100:]
dns_test = dns[-100:]

# print(dsa_train.shape)

batch_num = np.size(dsa_train, 0)
# print(batch_num)

loss = []
# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for step in range(num_steps):
        #         batch_x, batch_y = mnist.train.next_batch(batch_size)
        # batch_x = np.zeros((batch_size, 5))
        # batch_y = np.zeros((batch_size, 4))
        batch_x = dsa_train[(step*batch_size)%batch_num : (step*batch_size+batch_size)%batch_num, : ]
        batch_y = dns_train[(step*batch_size)%batch_num : (step*batch_size+batch_size)%batch_num, : ]
        # print((step*batch_size)%batch_num)
        # print((step*batch_size+batch_size-1)%batch_num)
        # print(batch_x.shape)
        # print(batch_y.shape)

        # Run optimization op (backprop)
        loss_train, _ = sess.run([loss_op, train_op], feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0:
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss_train) )
            loss.append(loss_train)

    print("Optimization Finished!")

    # test set
    loss_test = sess.run(loss_op, feed_dict={X: dsa_test, Y: dns_test})
    print("test loss is {}".format(loss_test))

    # plot training loss
    # plt.plot(loss)
    # plt.show()

    # Find the mapping between source and target
    mc2d_states = mc2d_env.observation_space.shape[0] # 2
    mc3d_states = mc3d_env.observation_space.shape[0] # 4
    mc2d_actions = mc2d_env.action_space.n # 3
    mc3d_actions = mc3d_env.action_space.n # 5

    mse_state_mappings = np.zeros((2,)*mc3d_states) # 2 by 2 by 2 by 2
    mse_action_mappings = np.ndarray(shape=(mc3d_actions,mc2d_actions, mc3d_states*mc3d_states)) # 5 by 3 by 16
    mse_action_mappings.fill(-1)

    state_count = 0
    for s0 in range(mc2d_states): # s0 is the first state of target states, x
        for s1 in range(mc2d_states): # s1 is second state of target states, y
            for s2 in range(mc2d_states):  # s2 is third state of target states, x_dot
                for s3 in range(mc2d_states):  # s3 is fourth state of target states, y_dot

                    state_losses = []

                    for a_mc3d in range(mc3d_actions):
                        for a_mc2d in range(mc2d_actions):
                            states = np.array([x[0] for x in dsource if x[1]==a_mc2d])
                            actions = np.array([x[1] for x in dsource if x[1] == a_mc2d])
                            n_states = np.array([x[2] for x in dsource if x[1]==a_mc2d])

                            if (states.size==0) or (actions.size==0) or (n_states.size==0):
                                print('this happened..') # TODO
                                # mse_action_mappings[a_mc3d, a_mc2d, state_count] = 0
                                continue

                            # transform to dsource_trans
                            actions_trans = np.ndarray(shape=(actions.size,))
                            actions_trans.fill(a_mc3d)
                            input_trans = np.array([states[:, s0], states[:, s1], states[:, s2], states[:, s3], actions_trans]).T
                            # input_trans = [states_trans, actions]
                            n_states_trans = np.array([n_states[:,s0], n_states[:,s1], n_states[:,s2], n_states[:,s3]]).T

                            # calculate mapping error
                            loss_mapping = sess.run(loss_op, feed_dict={X: input_trans, Y: n_states_trans})
                            # print('loss_mapping is {}'.format(loss_mapping))

                            state_losses.append(loss_mapping)
                            mse_action_mappings[a_mc3d, a_mc2d, state_count] = loss_mapping

                    mse_state_mappings[s0, s1, s2, s3] = np.mean(state_losses)
                    state_count += 1

    # mse_action_mappings_result = [[np.mean(mse_action_mappings[a_mc3d, a_mc2d, :]) for a_mc2d in range(mc2d_actions)] for a_mc3d in range(mc3d_actions)]

    mse_action_mappings_result = np.zeros((mc3d_actions, mc2d_actions))
    for a_mc3d in range(mc3d_actions):
        for a_mc2d in range(mc2d_actions):
            losses_act = []
            for s in range(mc3d_states*mc3d_states):
                if mse_action_mappings[a_mc3d, a_mc2d, s] != -1:
                    # print(mse_action_mappings[a_mc3d, a_mc2d, s])
                    losses_act.append(mse_action_mappings[a_mc3d, a_mc2d, s])
            mse_action_mappings_result[a_mc3d, a_mc2d] = np.mean(losses_act)

    print('action mapping: {}'.format(mse_action_mappings_result))
    print('state mapping {}'.format(mse_state_mappings))

    print('x,x,x,x: {}'.format(mse_state_mappings[0][0][0][0]))
    print('x,x,x,x_dot: {}'.format(mse_state_mappings[0][0][0][1]))
    print('x,x,x_dot,x: {}'.format(mse_state_mappings[0][0][1][0]))
    print('x,x,x_dot,x_dot: {}'.format(mse_state_mappings[0][0][1][1]))
    print('x,x_dot,x,x: {}'.format(mse_state_mappings[0][1][0][0]))
    print('x,x_dot,x,x_dot: {}'.format(mse_state_mappings[0][1][0][1]))
    print('x,x_dot,x_dot,x: {}'.format(mse_state_mappings[0][1][1][0]))
    print('x,x_dot,x_dot,x_dot: {}'.format(mse_state_mappings[0][1][1][1]))
    print('x_dot,x,x,x: {}'.format(mse_state_mappings[1][0][0][0]))
    print('x_dot,x,x,x_dot: {}'.format(mse_state_mappings[1][0][1][0]))
    print('x_dot,x,x_dot,x: {}'.format(mse_state_mappings[1][0][1][1]))
    print('x_dot,x,x_dot,x_dot: {}'.format(mse_state_mappings[1][1][0][0]))
    print('x_dot,x_dot,x,x: {}'.format(mse_state_mappings[1][0][0][1]))
    print('x_dot,x_dot,x,x_dot: {}'.format(mse_state_mappings[1][1][0][1]))
    print('x_dot,x_dot,x_dot,x: {}'.format(mse_state_mappings[1][1][1][0]))
    print('x_dot,x_dot,x_dot,x_dot: {}'.format(mse_state_mappings[1][1][1][1]))

    with open('./data/mse_state_mappings.pkl', 'wb') as file:
        pickle.dump(mse_state_mappings, file)

    with open('./data/mse_action_mappings.pkl', 'wb') as file:
        pickle.dump(mse_action_mappings, file)


    print("Done exporting MSE file")
