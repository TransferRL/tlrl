import tensorflow as tf
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections import deque

def huber_loss(predictions, labels, delta):
    """partially copied and modified from Tensorflow r1.4"""
    error = tf.subtract(predictions, labels)
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    # The following expression is the same in value as
    # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
    # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
    # This is necessary to avoid doubling the gradient, since there is already a
    # nonzero contribution to the gradient from the quadratic term.
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return losses
    

class q_network(object):
    
    """
    This class builds and runs a neural network for deep q learning
    """
    
    def __init__(self
                 ,discount_rate
                 ,mem_size
                 ,sample_size
                 ,n_input_units
                 ,n_output_units
                 ,n_hidden_layers = 2
                 ,n_hidden_units = 15
                 ,activation = tf.nn.relu
                 ,opt = tf.train.MomentumOptimizer
                 ,opt_kws = {'learning_rate':0.001,'momentum':0.5}
                 ):
        self.discount_rate = discount_rate
        self.mem_size = mem_size
        self.sample_size = sample_size
        self.n_input_units = n_input_units
        self.n_output_units = n_output_units
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.activation = activation
        self.opt = opt(**opt_kws)
        self.memory = deque([],mem_size)
        self.sess = None
        self.losses = []
        
    def initialize_graph(self):
        
        def weight_matrix(n_from,n_to):
            return tf.Variable(tf.truncated_normal(
                shape=(n_from,n_to)
                ,mean=0.0
                ,stddev=1/np.sqrt(n_from+n_to)
                ,dtype=tf.float32))

        def bias_matrix(n_to):
            return tf.Variable(tf.zeros(shape=(1,n_to),dtype=tf.float32))
        
        hidden_dict = {}
        
        input_layer = tf.placeholder(name='q_input', shape=(None,self.n_input_units),dtype=tf.float32)

        for n in range(1,self.n_hidden_layers+1):
            hidden_dict[n] = {}
            if n == 1:
                hidden_dict[n]['weights'] = weight_matrix(self.n_input_units,self.n_hidden_units)
                hidden_dict[n]['bias'] = bias_matrix(self.n_hidden_units)
                hidden_dict[n]['batch_norm'] = tf.contrib.layers.layer_norm(tf.matmul(input_layer, hidden_dict[n]['weights']) + hidden_dict[n]['bias'], center=True, scale=True)
                hidden_dict[n]['layer'] = self.activation(hidden_dict[n]['batch_norm'])
            else:
                hidden_dict[n]['weights'] = weight_matrix(self.n_hidden_units,self.n_hidden_units)
                hidden_dict[n]['bias'] = bias_matrix(self.n_hidden_units)
                hidden_dict[n]['batch_norm'] = tf.contrib.layers.layer_norm(tf.matmul(hidden_dict[n-1]['layer'],hidden_dict[n]['weights']) + hidden_dict[n]['bias'], center=True, scale=True)
                hidden_dict[n]['layer'] = self.activation(hidden_dict[n]['batch_norm'])


        output_weights = weight_matrix(self.n_hidden_units,self.n_output_units)
        output_bias = bias_matrix(self.n_output_units)

        output_pred = tf.matmul(hidden_dict[self.n_hidden_layers]['layer'],output_weights) + output_bias
        output_truth = tf.placeholder(shape=(None,self.n_output_units),dtype=tf.float32)

        loss = tf.reduce_sum(tf.reduce_mean((1/2)*((output_pred-output_truth)**2),axis=0))
        
        all_variables = [hidden_dict[n]['weights'] for n in range(1,self.n_hidden_layers+1)] + [output_weights] + [hidden_dict[n]['bias'] for n in range(1,self.n_hidden_layers+1)] + [output_bias]
        opt_op = self.opt.minimize(loss, var_list = all_variables)
        
        self.input_layer = input_layer
        self.output_pred = output_pred
        self.output_truth = output_truth
        self.loss = loss
        self.all_variables = all_variables
        self.opt_op = opt_op
    
    def open_session(self):
        self.sess = tf.Session()
        
    def close_session(self):
        self.sess.close()
        
    def initialize_new_variables(self):
        self.sess.run(tf.global_variables_initializer())
        
    def run_training(self,n_epochs,states,coded_actions,transitions,rewards):
        """
            'states' are the vectors of s as row vectors in matrix of samples.
            'coded_actions' are a single 1-D vector of actions, encoded as index of
        one-hot vector of actions, i.e. if one-hot = [0,1,0,0] then coded_output = 1.
        This hack allows for easy generation of ground_truth samples. 
            'transitions' are vectors of s' as row vectors in matrix of samples.
            'rewards' are a single 1-D vector of rewards.
        """
        pbar = tqdm(range(n_epochs))
        _losses = []
        for _ in pbar:

            bellman_trans_q = np.max(self.sess.run(self.output_pred, feed_dict= {self.input_layer:transitions}) ,axis=1)

            ground_truth = self.sess.run(self.output_pred, feed_dict={self.input_layer:states})
            #ground_truth = np.zeros(shape=(states.shape[0],self.output_pred.shape[1]),dtype=float)
            
            ground_truth[list(range(len(states))),coded_actions] = rewards + (self.discount_rate * bellman_trans_q) #this is the bellman update for fitted q iteration, with all non-action outputs as the Q value that will be predicted by the current network - this negates any back-prop through these output nodes. 

            _, _loss = self.sess.run([self.opt_op,self.loss],feed_dict={self.input_layer:states, self.output_truth:ground_truth})
            self.losses.append(_loss)
            pbar.set_description('loss: {}'.format(_loss))
            
    def plot_loss(self):
        plt.plot(list(range(len(self.losses))),self.losses)
        plt.title('DQN loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
        
    def add_new_obvs(self,states, actions, transitions, rewards):
        for i in range(len(states)):
            self.memory.append([states[i], actions[i], transitions[i], rewards[i]])
            
    def get_memory_sample(self, size):
        states, actions, transitions, rewards = [], [], [], []
        for i in np.random.choice(range(len(self.memory)),size):
            states.append(self.memory[i][0])
            actions.append(self.memory[i][1])
            transitions.append(self.memory[i][2])
            rewards.append(self.memory[i][3])
            
        return np.array(states).astype(float), np.array(actions), np.array(transitions).astype(float), np.array(rewards)
    
    def get_next_action(self,state):
        return np.argmax(self.sess.run(self.output_pred, feed_dict={self.input_layer:state}) ,axis=1)
            
        
        
if __name__ == '__main__':
    
    _n_samples = 20000
    _n_input = 10
    _n_output = 5

    actions = np.random.binomial(_n_output-1,[1/_n_output],_n_samples)
    states = np.random.randn(_n_input*_n_samples).reshape(_n_samples,_n_input).astype(float)
    transitions = np.random.randn(_n_input*_n_samples).reshape(_n_samples,_n_input).astype(float)
    rewards = np.random.binomial(1,[1/5],_n_samples).reshape(1,-1)

    dqn = q_network(discount_rate = 0.9
                 ,mem_size = 5000
                 ,sample_size = 1000
                 ,n_input_units = _n_input
                 ,n_output_units = _n_output
                 ,n_hidden_layers = 3
                 ,n_hidden_units = 64
                 ,activation = tf.nn.relu
                 ,opt = tf.train.MomentumOptimizer
                 ,opt_kws = {'learning_rate':0.0001,'momentum':0.2}
                 )
    
    dqn.initialize_graph()
    dqn.open_session()
    dqn.initialize_new_variables()
    
    dqn.add_new_obvs(states, actions, transitions, rewards)
    _states, _actions, _transitions, _rewards = dqn.get_memory_sample(dqn.mem_size)
    dqn.run_training(500, _states, _actions, _transitions, _rewards)
    dqn.plot_loss()