from dqn_baseline import main
import tensorflow as tf
from envs import ENVS_DICTIONARY


params_dictionary = {}
params_dictionary["discount_rate"] = 1.0
params_dictionary["mem_size"] = 50000
params_dictionary["sample_size"] = 32
params_dictionary["n_hidden_layers"] = 1
params_dictionary["n_hidden_units"] = 32
params_dictionary["activation"] = tf.nn.relu
params_dictionary["optimizer"] = tf.train.AdamOptimizer
params_dictionary["opt_kws"] = {'learning_rate':0.01}
params_dictionary["n_episodes"] = 200
params_dictionary["n_epochs"] = 1
params_dictionary["retrain_period"] = 1
params_dictionary["epsilon"] = 0.15
params_dictionary["epsilon_decay"] = 0.99
params_dictionary["ini_steps_retrain"] = 1000

main(ENVS_DICTIONARY['2DMountainCar'],'2DMC',2,3,params_dictionary)
