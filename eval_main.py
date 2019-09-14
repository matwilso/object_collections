#!/usr/bin/env python3
import itertools
import time
import os
import random
import pickle
import copy
import shutil

import numpy as np
import tensorflow as tf
from anyrl.utils.tf_state import load_vars, save_vars
from gym.wrappers.time_limit import TimeLimit
import quaternion

import object_collections.rl.util as rl_util
from object_collections.define_flags import FLAGS
from object_collections.envs import make_fn
from object_collections.rl.agents import Scripted
from object_collections.rl.encoders import DYN
from object_collections.sl.trainer import Trainer as SL_Trainer
from object_collections.rl.trainer import OnPolicyTrainer, OffPolicyTrainer
import scipy
from object_collections.envs.utils.table import TABLES
cos_dist = scipy.spatial.distance.cosine

Trainer = {'scripted': OnPolicyTrainer, 'sac': OffPolicyTrainer}[FLAGS['agent']]

"""
Used to evaluate models in simulation
"""

def eval_policy():
    import matplotlib.pyplot as plt
    FLAGS['num_envs'] = 1
    FLAGS['explore_anneal'] = 0
    FLAGS['explore_frac'] = 0
    FLAGS['phi_noise'] = 0.0
    FLAGS['is_training'] = False

    trainer = Trainer(FLAGS)
    sess = trainer.sess
    env = trainer.env

    successes = 0
    failures = 0

    obs = env.reset()
    while True:
        model_out = trainer.agent.model.step([obs], None, mode='explore')
        obs, rew, done, info = env.step(model_out['actions'][0])

        vec_obs = trainer.obs_vectorizer.to_vecs([obs])
        if FLAGS['aac']:
            dyn_vals = trainer.value_encoder.step({'s': vec_obs, 'a': model_out['actions'], 's_next':vec_obs})
        else:
            dyn_vals = trainer.encoder.step({'s': vec_obs, 'a': model_out['actions'], 's_next':vec_obs})
        dist = cos_dist(dyn_vals['phi_s'][0], dyn_vals['phi_g'][0])

        img = trainer.obs_vectorizer.to_np_dict(vec_obs)['image'][0]
        goal_img = trainer.obs_vectorizer.to_np_dict(vec_obs)['goal_image'][0]

        arr = trainer.obs_vectorizer.to_np_dict(vec_obs)['array'][0]
        goal_arr = trainer.obs_vectorizer.to_np_dict(vec_obs)['goal_array'][0]

        carr = arr * (0.5*TABLES[FLAGS['default_table']]['wood'][:2])
        cgarr = goal_arr * (0.5*TABLES[FLAGS['default_table']]['wood'][:2])

        mx, my = np.mean(cgarr, axis=0)
        in_box = True
        for elem in carr:
            ax, ay = elem
            if mx - 0.125 < ax and ax < mx + 0.125 and my - 0.125 < ay and ay < my + 0.125:
                in_box = in_box and True
            else:
                in_box = False

        if in_box:
            successes += 1
            obs = env.reset()
            print('rate = {}/{} = {}'.format(successes, successes+failures, successes / (successes + failures)))
        elif done:
            failures += 1
            obs = env.reset()
            print('rate = {}/{} = {}'.format(successes, successes+failures, successes / (successes + failures)))

        if successes + failures == 100:
            print(FLAGS['suffix'], FLAGS['reset_mode'])
            break

def _to_obs_vec(obs):
    def subdict(dict, keys): return {key: dict[key] for key in keys}
    return copy.deepcopy(tuple(subdict(obs, sorted(['array', 'single'])).values()))

def main():
    random.seed(FLAGS['seed'])
    np.random.seed(FLAGS['seed'])
    tf.set_random_seed(FLAGS['seed'])
    eval_policy()

if __name__ == '__main__':
    main()
