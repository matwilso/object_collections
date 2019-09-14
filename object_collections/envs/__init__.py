import numpy as np
from gym.wrappers.time_limit import TimeLimit
import os
if 'no_mujoco' not in os.environ:
    import mujoco_py
    from .object_env import ObjectCollectionEnv
from .utils.table import TABLES
from .utils.common import discretize, undiscretize, map_continuous, unmap_continuous

def make_fn(seed, FLAGS):
    def _thunk():
        env = TimeLimit(ObjectCollectionEnv(FLAGS), max_episode_steps=FLAGS['max_episode_steps'])
        env.seed(seed)
        return env
    return _thunk
