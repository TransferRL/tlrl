from dqn_baseline import main
import tensorflow as tf
from envs import ENVS_DICTIONARY

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           10000, 0.0, staircase=True)

params_dictionary = {}
params_dictionary["discount_rate"] = 0.9
params_dictionary["mem_size"] = 1
params_dictionary["sample_size"] = 1
params_dictionary["n_hidden_layers"] = 1
params_dictionary["n_hidden_units"] = 32
params_dictionary["activation"] = tf.nn.relu
params_dictionary["optimizer"] = tf.train.MomentumOptimizer
params_dictionary["opt_kws"] = {'learning_rate':learning_rate,'momentum':0.5}
params_dictionary["n_episodes"] = 500
params_dictionary["n_epochs"] = 1
params_dictionary["retrain_period"] = 1
params_dictionary["epsilon"] = 0.5
params_dictionary["epsilon_decay"] = 0.999
params_dictionary["ini_steps_retrain"] = 1

main(ENVS_DICTIONARY['3DMountainCar'](),'3DMC',4,5,params_dictionary,_3d=True)