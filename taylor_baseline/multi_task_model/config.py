class AgentConfig(object):
  scale = 1000
  display = False

  max_step = 5000 * scale
  memory_size = 100 * scale

  batch_size = 32
  random_start = 30
  cnn_format = 'NCHW'
  discount = 1
  target_q_update_step = 1 * scale
  learning_rate = 0.0025
  learning_rate_minimum = 0.0025
  learning_rate_decay = 0.96
  learning_rate_decay_step = 500 * scale

  ep_end = 0.1
  ep_start = 1.
  ep_end_t = memory_size

  history_length = 1
  train_frequency = 4
  learn_start = 5. * scale

  min_delta = -1
  max_delta = 1

  double_q = False
  dueling = False

  _test_step = 5 * scale
  _save_step = _test_step * 10

class EnvironmentConfig(object):
  max_reward = -1.
  min_reward = -10000.

class DQNConfig(AgentConfig, EnvironmentConfig):
  model = ''
  pass

class M1(DQNConfig):
  backend = 'tf'
  env_type = '3D'
  action_repeat = 1
  env_name = 'ThreeDMountainCar-v0'

  # Network Parameters
  n_hidden_1 = 32  # 1st layer number of neurons
  n_hidden_2 = 32  # 2nd layer number of neurons
  num_input = 4
  num_output = 5

  state_width  = 4
  state_height = 1

class M2(DQNConfig):
  backend = 'tf'
  env_type = '2D'
  env_name = 'MountainCar-v0'
  action_repeat = 1
  # Network Parameters
  n_hidden_1 = 32  # 1st layer number of neurons
  n_hidden_2 = 32  # 2nd layer number of neurons
  num_input = 2
  num_output = 3

  state_width = 2
  state_height = 1


def get_config(FLAGS):
  if FLAGS.model == 'm1':
    config = M1
  elif FLAGS.model == 'm2':
    config = M2

  for k, v in FLAGS.__dict__['__flags'].items():
    if k == 'gpu':
      if v == False:
        config.cnn_format = 'NHWC'
      else:
        config.cnn_format = 'NCHW'

    if hasattr(config, k):
      setattr(config, k, v)

  return config
