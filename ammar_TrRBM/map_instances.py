import numpy as np
import tensorflow as tf
import trrbm
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


def load_samples(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def unpack_samples(samples):
    unpacked = []
    actions = []
    for sample in samples:
        state = np.array(sample[0])
        state_prime = np.array(sample[2])
        reward = np.array(sample[3])
        terminal = np.array(int(sample[4]))
        
        # need to one-hot-encode the action 
        action = [sample[1]]
        
        unpacked.append(np.concatenate([state,state_prime]))
        actions.append(action)
        
    actions = OneHotEncoder(sparse=False).fit_transform(np.stack(actions)).astype(float)
    return np.concatenate([unpacked,actions],axis=1)
    
def even_out_samplesizes(samples1, samples2):
    if len(samples1) < len(samples2):
        samples1 = np.tile(samples1,[np.math.ceil(len(samples2)/(len(samples1))),1])
        np.random.shuffle(samples1)
        samples1 = samples1[:len(samples2)]
        
    if len(samples1) > len(samples2):
        samples2 = np.tile(samples2,[np.math.ceil(len(samples1)/(len(samples2))),1])
        np.random.shuffle(samples2)
        samples2 = samples2[:len(samples1)]
        
    return samples1, samples2

def standardize_samples(samples):
    return StandardScaler().fit_transform(samples)
    
def main():
    
    source_random_path = '../taylor_master/data/2d_instances.pkl'
    target_random_path = '../taylor_master/data/3d_instances.pkl'
    source_optimal_path = '../taylor_master/data/2d_instances.pkl'

    # load source task random samples
    source_random = unpack_samples(load_samples(source_random_path))

    # load target task random samples
    target_random = unpack_samples(load_samples(target_random_path))
    
    # prepare samples
    source_random, target_random = even_out_samplesizes(source_random, target_random)
    s

    # load the model

    rbm = trrbm.RBM(
        name = "TrRBM",
        v1_size = source_random.shape[1], 
        h_size = 100, 
        v2_size = target_random.shape[1], 
        n_data = source_random.shape[0], 
        batch_size = 100, 
        learning_rate = 0.001,
        num_epochs = 500, 
        n_factors = 40,
        k = 1,
        use_tqdm = True,
        show_err_plt = True
    )

    # train TrRBM model
    errs = rbm.train(source_random, target_random)
    
    if rbm.show_err_plt:
        plt.plot(range(len(rbm.cost)), rbm.cost)
        plt.title('training reconstruction error')
        plt.xlabel('epoch')
        plt.ylabel('avg reconstruction error')
        plt.show()

    # load source task optimal instances
    source_optimal = 
    

    # map to target instances
    
if __name__ == '__main__':
    main()