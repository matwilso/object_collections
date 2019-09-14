import os
import numpy as np
import tensorflow as tf
import yaml

import object_collections.rl.util as rl_util
from object_collections.define_flags import ACTIVS, FLAGS, INITS, post_load_flags
from object_collections.sl.building_blocks import MDN_Head
from object_collections.sl.trainer import Trainer as SL_Trainer
from object_collections.rl.agents import Scripted
from object_collections.rl. import DYN
from object_collections.rl.trainer import Trainer

"""
Script used to Frankenstein some trained models 
(like putting a CNN and GN into a single checkpoint so they can be loaded by the RL Trainer)
"""

action_dist = rl_util.convert_to_dict_dist(FLAGS['action_space'].spaces)
obs_vectorizer = rl_util.convert_to_dict_dist(FLAGS['observation_space'].spaces)

in_batch_shape = (None,) + obs_vectorizer.out_shape

sas_phs = {}
sas_phs['s'] = obs_ph = tf.placeholder(tf.float32, shape=in_batch_shape, name='obs_ph')
sas_phs['a'] = actions_ph = tf.placeholder(tf.float32, (None,) + action_dist.out_shape, name='actions_ph')  # actions that were taken
sas_phs['s_next'] = next_obs_ph = tf.placeholder(tf.float32, shape=in_batch_shape, name='next_obs_ph')
sas_phs['r'] = tf.placeholder(tf.float32, shape=None, name='reward_ph')
sas_phs['d'] = tf.placeholder(tf.float32, shape=None, name='done_ph')
sas_vals = rl_util.sarsd_to_vals(sas_phs, obs_vectorizer, FLAGS)

tf.trainable_variables()
sess = tf.InteractiveSession()

# load environment variable for where your cnn weights were saved
cnn_path = os.environ['cnn_path']

def get_filename(fname):
    dirname = os.path.dirname(fname)
    if abs(len(dirname) - len(fname)) <= 5:
        return tf.train.latest_checkpoint(dirname)
    else:
        return fname

cnn_ckpt = get_filename(cnn_path)
print(cnn_path)

# load environment variable for where you want the output of this checkpoint saver to go
out_path = os.environ['out_path']
print(out_path)


pi_vars = []

# This is for a model trained with dyn_coadapt
# do a check. sometimes we don't need to rename if it is already ok
names = [name for name,_ in tf.contrib.framework.list_variables(cnn_ckpt)]  
is_conv_dyn = False
for name in names: 
    is_conv_dyn = is_conv_dyn or 'Conv_DYN' in name
print('is_conv_dyn', is_conv_dyn)

for var_name, _ in tf.contrib.framework.list_variables(cnn_ckpt):
    if ('DYN_Model' in var_name or 'MDN' in var_name) and 'Adam' not in var_name:
        np_var = tf.contrib.framework.load_variable(cnn_ckpt, var_name)
        if is_conv_dyn:
            if 'Conv_DYN' not in var_name and ('GraphFormer' in var_name or 'DYN_Model/linear/' in var_name):
                #new_name = var_name.replace('DYN/', 'GoalDYN/')
                new_name = var_name.replace('DYN/', 'ValueDYN/')
            elif 'Conv_DYN' in var_name:
                new_name = var_name.replace('Conv_DYN', 'DYN')
            else:
                new_name = var_name
        else:
            new_name = var_name
        tf_var = tf.Variable(np_var, name=new_name)
        pi_vars.append(tf_var)
    # handle VAE mode
    elif 'ConvVAE' in var_name and 'Adam' not in var_name:
        np_var = tf.contrib.framework.load_variable(cnn_ckpt, var_name)
        tf_var = tf.Variable(np_var, name=var_name)
        pi_vars.append(tf_var)

print('Grabbed CNN vars from', cnn_path)

try:
    # TODO: some type of assertion that these came from the same origin.
    # could check load_path same, but there are cases that this does not cover

    rl_path = os.environ['rl_path']
    rl_ckpt = get_filename(rl_path)
    print(rl_path)
    
    for var_name, _ in tf.contrib.framework.list_variables(rl_ckpt):
        if 'SAC' in var_name and 'Adam' not in var_name:
            np_var = tf.contrib.framework.load_variable(rl_ckpt, var_name)
            tf_var = tf.Variable(np_var, name=var_name)
            pi_vars.append(tf_var)
    print('Grabbed RL vars from ', rl_path)
except Exception as e:
    print('NO RL', e)

try:
    value_path = os.environ['value_path']
    value_ckpt = get_filename(value_path)
    print('Value', value_path)

    for var_name, _ in tf.contrib.framework.list_variables(value_ckpt):
        if ('DYN_Model' in var_name or 'MDN' in var_name) and 'Adam' not in var_name and 'Conv_DYN' not in var_name:
            np_var = tf.contrib.framework.load_variable(value_ckpt, var_name)
            new_name = var_name.replace('DYN/', 'ValueDYN/')
            tf_var = tf.Variable(np_var, name=new_name)
            pi_vars.append(tf_var)
        if 'VAE' in var_name and 'Adam' not in var_name:
            np_var = tf.contrib.framework.load_variable(value_ckpt, var_name)
            new_name = var_name.replace('VAE/', 'ValueVAE/')
            tf_var = tf.Variable(np_var, name=new_name)
            pi_vars.append(tf_var)

    print('Grabbed Value vars from', value_ckpt)

except Exception as e:
    print('NO Value', e)



dumper = tf.train.Saver(var_list=pi_vars)
sess.run(tf.variables_initializer(pi_vars))
dumper.save(sess, out_path)

[print(var) for var in tf.contrib.framework.list_variables(out_path)]