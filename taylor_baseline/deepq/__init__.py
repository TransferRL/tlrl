from baselines.deepq import models  # noqa
from baselines.deepq.build_graph import build_train  # noqa
from deepq.build_graph import build_q_values, build_act
from deepq.simple import learn, load  # noqa
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=True)