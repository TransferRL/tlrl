import numpy as np

class History:
  def __init__(self, config):
    self.cnn_format = config.cnn_format

    batch_size, history_length, state_height, state_width = \
        config.batch_size, config.history_length, config.state_height, config.state_width

    self.history = np.zeros(
        [history_length, state_height, state_width], dtype=np.float32)

  def add(self, state):
    self.history[:-1] = self.history[1:]
    self.history[-1] = state

  def reset(self):
    self.history *= 0

  def get(self):
    if self.cnn_format == 'NHWC':
      return np.transpose(self.history, (1, 2, 0))
    else:
      return self.history
