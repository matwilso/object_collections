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
cos_dist = scipy.spatial.distance.cosine

Trainer = {'scripted': OnPolicyTrainer, 'sac': OffPolicyTrainer}[FLAGS['agent']]

def play():
    import matplotlib.pyplot as plt
    FLAGS['num_envs'] = 1
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
        #dyn_vals = trainer.dyn.step({'s': vec_obs, 'a': model_out['actions'], 's_next':vec_obs})
        if FLAGS['aac']:
            dyn_vals = trainer.value_encoder.step({'s': vec_obs, 'a': model_out['actions'], 's_next':vec_obs})
        else:
            dyn_vals = trainer.encoder.step({'s': vec_obs, 'a': model_out['actions'], 's_next':vec_obs})
        dist = cos_dist(dyn_vals['phi_s'][0], dyn_vals['phi_g'][0])
        #print('dist', dist)

        img = trainer.obs_vectorizer.to_np_dict(vec_obs)['image'][0]
        goal_img = trainer.obs_vectorizer.to_np_dict(vec_obs)['goal_image'][0]

        arr = trainer.obs_vectorizer.to_np_dict(vec_obs)['array'][0]
        goal_arr = trainer.obs_vectorizer.to_np_dict(vec_obs)['goal_array'][0]

        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        ax1.imshow(img)
        ax2.imshow(goal_img)
        ax3.scatter(arr[:,0], arr[:,1])
        ax3.scatter(goal_arr[:,0], goal_arr[:,1])
        ax3.set_ylim(-1,1)
        ax3.set_xlim(-1,1)
        plt.title(str(dist))
        plt.savefig('./vids/'+str(time.time()) + '.png')

        if dist < FLAGS['goal_threshold']:
            successes += 1
            #print('SUCCESS')
            obs = env.reset()
            print('rate = {}/{} = {}'.format(successes, successes+failures, successes / (successes + failures)))
            #shutil.rmtree('./vids/')
        elif done:
            failures += 1
            #print('FAILURE')
            obs = env.reset()
            print('rate = {}/{} = {}'.format(successes, successes+failures, successes / (successes + failures)))
            #shutil.rmtree('./vids/')
        #if done: break

def trainer():
    with rl_util.Timer('trainer build', FLAGS):
        t = Trainer(FLAGS)
    out = t.run()
    print(out)

def scripted_test():
    env = make_fn(FLAGS['seed'], FLAGS)()
    obs = env.reset()

    action_dist = rl_util.convert_to_dict_dist(env.action_space.spaces)
    obs_vectorizer = rl_util.convert_to_dict_dist(env.observation_space.spaces)

    sess = tf.InteractiveSession()

    # sarsd phs
    in_batch_shape = (None,) + obs_vectorizer.out_shape
    sarsd_phs = {}
    sarsd_phs['s'] = tf.placeholder(tf.float32, shape=in_batch_shape, name='s_ph')
    sarsd_phs['a'] = tf.placeholder(tf.float32, (None,) + action_dist.out_shape, name='a_ph')  # actions that were taken
    sarsd_phs['s_next'] = tf.placeholder(tf.float32, shape=in_batch_shape, name='s_next_ph')
    sarsd_phs['r'] = tf.placeholder(tf.float32, shape=(None,), name='r_ph')
    sarsd_phs['d'] = tf.placeholder(tf.float32, shape=(None,), name='d_ph')
    sarsd_vals = rl_util.sarsd_to_vals(sarsd_phs, obs_vectorizer, FLAGS)

    scri = Scripted(sas_vals=sarsd_vals, sas_phs=sarsd_phs, embed_phs={}, action_dist=action_dist, obs_vectorizer=obs_vectorizer, FLAGS=FLAGS, dyn_model=None)

    for i in itertools.count(start=1):
        model_out = scri.model.step([obs], None)
        obs, reward, done, info = env.step(model_out['actions'][0])
        #print(reward)
        if done:
            obs = env.reset()

def random_policy():
    env = make_fn(FLAGS['seed'], FLAGS)()
    obs = env.reset()
    for i in itertools.count(start=1):
        obs, reward, done, info = env.step(env.unwrapped.random_action())
        if done:
            obs = env.reset()

def _to_obs_vec(obs):
    def subdict(dict, keys): return {key: dict[key] for key in keys}
    return copy.deepcopy(tuple(subdict(obs, sorted(['array', 'single'])).values()))

def main():
    random.seed(FLAGS['seed'])
    np.random.seed(FLAGS['seed'])
    tf.set_random_seed(FLAGS['seed'])

    if FLAGS['scripted_test']:
        scripted_test()
    elif FLAGS['random_policy']:
        random_policy()
    elif FLAGS['play']:
        print()
        print('PLAYING')
        print()
        play()
    else:
        trainer()

if __name__ == '__main__':
    main()
