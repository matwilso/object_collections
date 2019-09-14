import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
from anyrl.algos.advantages import GAE
from anyrl.models import TFActorCritic

import sonnet as snt
import object_collections.rl.util as rl_util
from graph_nets import utils_tf
from object_collections.envs import discretize, map_continuous, unmap_continuous
from object_collections.envs.utils.table import TABLES
from object_collections.sl.aux_losses import (mask_state, mdn_loss, mdn_metrics)
                                      
from object_collections.sl.building_blocks import (GraphFormer, Transformer)
from object_collections.rl.models import SACAC
from tensorflow_probability import distributions as tfd
from object_collections.tf_util import get_session

def subdict(dict, keys): return {key: dict[key] for key in keys}
def prefix_vals(name, vals): return {name.lower() + '_' + key: vals[key] for key in vals} # add a prefix to dictionary keys


class Agent(object):
    init_op = tf.no_op()

class ScriptedModel(object):
    """
    Algorithm used to compute actions for scripted policy

    This is in a structure to match the learned agent API
    """
    def __init__(self, sess, sas_phs, action_dist, obs_vectorizer, FLAGS):
        self.action_dist = action_dist
        self.obs_vectorizer = obs_vectorizer
        self.FLAGS = FLAGS

    @property
    def stateful(self):
        return False

    def start_state(self, batch_size):
        return None

    def step(self, observations, states, mode=None):
        model_outs = defaultdict(lambda: [])

        obses = self.obs_vectorizer.to_vecs(observations)
        obses = self.obs_vectorizer.to_np_dict(obses)

        for i in range(len(observations)): 
            obs = {key: obses[key][i] for key in obses}
            action = {}
            cubes = obs['array'][:,:2]
            cubes *= (0.5*TABLES[self.FLAGS['default_table']]['wood'][:2])

            #cube = obs['single'][np.random.randint(len(obs['single']))]
            cube = cubes[np.random.randint(self.FLAGS['num_objects'])]

            if self.FLAGS['mixed_script']:
                def xy_d(): return np.exp(np.random.uniform(np.log(0.03), np.log(0.8*self.FLAGS['max_dx'])))
            else:
                def xy_d(): return np.exp(np.random.uniform(np.log(0.03), np.log(0.7*self.FLAGS['max_dx'])))

            def neg_pos(): return (1 - 2*np.random.binomial(1, 0.5))

            dx = neg_pos() * xy_d()
            dy = neg_pos() * xy_d()

            # Check if it is near the edge.  If so, never push in that edge direction
            edgex = np.abs(cube[0]) > 0.15
            edgey = np.abs(cube[1]) > 0.50
            if edgex:
                dx = np.abs(dx) * np.sign(cube[0])
            if edgey:
                dy = np.abs(dy) * np.sign(cube[1])

            x = cube[0] + dx
            y = cube[1] + dy

            if self.FLAGS['discrete']:
                action['x'] = discretize('x', x, self.FLAGS)
                action['y'] = discretize('y', y, self.FLAGS)
                #print('x', x, self.FLAGS['ACTS']['x'][action['x']], 'y', y, self.FLAGS['ACTS']['y'][action['y']])
                gx = self.FLAGS['ACTS']['x'][action['x']]
                gy = self.FLAGS['ACTS']['y'][action['y']]

                dx = gx - cube[0]
                dy = gy - cube[1]
                #print('dx', dx, 'dy', dy)
            else:
                action['x'] = x
                action['y'] = y

            yaw = np.arctan(dy/dx)
            #print('yaw', yaw)

            if (dx > 0 and dy > 0):
                yaw -= np.pi
            elif (dx > 0 and dy < 0):
                yaw += np.pi

            if self.FLAGS['discrete']:
                action['yaw'] = discretize('yaw', yaw, self.FLAGS)# + (np.random.binomial(1, 0.10) * neg_pos()) # 10% chance to go either up or down
            else:
                action['yaw'] = yaw
            #print('ayaw', action['yaw'])
            if self.FLAGS['use_dist']:
                if self.FLAGS['discrete']:
                    action['dist'] = np.random.randint(0, self.FLAGS['act_dist_n'])
                else:
                    if self.FLAGS['mixed_script']:
                        action['dist'] = np.random.uniform(0.1*self.FLAGS['max_dx'], self.FLAGS['max_dx'])
                    else:
                        action['dist'] = np.random.uniform(0.5*self.FLAGS['max_dx'], self.FLAGS['max_dx'])
                    #action['dist'] = self.FLAGS['max_dx']

            if not self.FLAGS['discrete']:
                for key in action: action[key] = map_continuous(key, action[key], self.FLAGS)

            #action['dist'] = np.random.randint(self.FLAGS['act_dist_n'])
            #action['dist'] = discretize('dist', xy_d(), self.FLAGS) # this gets added in the env

            action_tuple = tuple([action[key] for key in sorted(action.keys())])

            model_outs['action_params'] += [0.0]
            model_outs['actions'] += [action_tuple]
            model_outs['values'] += [0.0]

        model_outs['states'] = None
        return dict(model_outs)

    def batch_outputs(self):
        return None

class Scripted(Agent):
    """Scripted"""
    def __init__(self, *,
                 sas_vals,
                 sas_phs,
                 embed_phs,
                 action_dist,
                 obs_vectorizer,
                 FLAGS,
                 name='Scripted',
                 **kwargs
                 ):

        self.adv_est = GAE(lam=0.95, discount=0.99) # never used, but to match interface
        self.name = name
        with tf.variable_scope(self.name):
            self.FLAGS = FLAGS
            self.model = ScriptedModel(None, sas_phs, action_dist, obs_vectorizer, self.FLAGS)

            self.phs = {}
            self.vals = {}
            self.phs['actions'] = sas_phs['a']
            self.vals['entropy'] = tf.reduce_mean(action_dist.entropy(self.phs['actions']))
            self.vals['loss'] = tf.constant(0.0, dtype=tf.float32)
        self.train_op = {}
        self.train_op['burn_in'] = self.train_op['post_burn'] = tf.no_op()

    def feed_dict(self, rollouts, batch, advantages, targets):
        actions = rl_util.select_model_out_from_batch('actions', rollouts, batch)
        feed_vals = {}
        feed_vals['actions'] = self.model.action_dist.to_vecs(actions)
        feed_dict = {self.phs[key]: feed_vals[key] for key in feed_vals}
        return feed_dict

class SAC(Agent):
    """SAC"""
    def __init__(self,
                 sas_vals,
                 sas_phs,
                 embed_phs,
                 action_dist,
                 obs_vectorizer,
                 FLAGS,
                 name='SAC',
                 dyn=None,
                 value_dyn=None,
                 act_limit=1.0,
                 gamma=0.99,
                 is_training_ph=None,
                 **kwargs
                 ):
        self.name = name
        self.FLAGS = FLAGS
        # TODO: move optimization stuff in here?

        with tf.variable_scope(self.name):
            self.model = SACAC(get_session(), sas_vals, sas_phs, embed_phs, action_dist, obs_vectorizer, FLAGS, dyn=dyn, value_dyn=value_dyn, act_limit=act_limit, is_training_ph=is_training_ph)

            self.phs = {}
            self.phs.update(sas_phs)
            self.embed_phs = embed_phs

            main_out, target_out = self.model.batch_outputs()

            # SAC optimization (taken from spinningup repo)
            # Min Double-Q
            min_q_pi = tf.minimum(main_out['q1_pi'], main_out['q2_pi'])
            # Targets for Q and V regression
            q_backup = tf.stop_gradient(self.phs['r'] + gamma * (1- self.phs['d']) * target_out['v'])
            v_backup = tf.stop_gradient(min_q_pi - self.FLAGS['sac_alpha'] * main_out['logp_pi'])

            # SAC losses
            pi_loss = tf.reduce_mean(self.FLAGS['sac_alpha'] * main_out['logp_pi'] - main_out['q1_pi'])
            q1_loss = 0.5 * tf.reduce_mean((q_backup - main_out['q1'])**2)
            q2_loss = 0.5 * tf.reduce_mean((q_backup - main_out['q2'])**2)
            v_loss = 0.5 * tf.reduce_mean((v_backup - main_out['v'])**2)
            value_loss = q1_loss + q2_loss + v_loss

            # Separate optimizers for pi and value
            self.pi_step = tf.get_variable('sac_pi_step', initializer=tf.constant(1, dtype=tf.int32), trainable=False)
            self.value_step = tf.get_variable('sac_value_step', initializer=tf.constant(1, dtype=tf.int32), trainable=False)
            pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.FLAGS['lr'])
            value_optimizer = tf.train.AdamOptimizer(learning_rate=self.FLAGS['lr'])

            value_vars = self.model.main_model._q1_net.get_variables() + self.model.main_model._q2_net.get_variables() + self.model.main_model._v_net.get_variables()
            pi_vars = self.model.main_model._pi_net.get_variables()

            # Policy train op 
            # (has to be separate from value train op, because q1_pi appears in pi_loss)
            train_pi_op = pi_optimizer.minimize(pi_loss, var_list=pi_vars, global_step=self.pi_step)

            # Value train op
            # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
            with tf.control_dependencies([train_pi_op]):
                train_value_op = value_optimizer.minimize(value_loss, var_list=value_vars, global_step=self.value_step)

            # Polyak averaging for target variables
            # (control flow because sess.run otherwise evaluates in nondeterministic order)
            with tf.control_dependencies([train_value_op]):
                target_update = tf.group([tf.assign(v_targ, self.FLAGS['polyak']*v_targ + (1-self.FLAGS['polyak'])*v_main)
                                          for v_main, v_targ in zip(self.model.main_model._v_net.get_variables(),
                                          self.model.target_model._v_net.get_variables())])

            # Initializing targets to match main variables
            target_init = tf.group([tf.assign(v_targ, v_main) for v_main, v_targ in zip(self.model.main_model._v_net.get_variables(), self.model.target_model._v_net.get_variables())])

            self.init_op = target_init

            # All ops to call during one training step
            self.main_vals = {'pi_loss': pi_loss, 'q1_loss': q1_loss, 'q2_loss': q2_loss, 'v_loss': v_loss}
            self.main_vals.update(main_out)
            self.main_vals.update({'train_pi_op': train_pi_op, 'train_value_op': train_value_op, 'target_update': target_update})

    def feed_dict(self, batch):
        feed_vals = batch
        if self.FLAGS['use_embed']:
            feed_dict = {self.embed_phs[key]: feed_vals[key] for key in feed_vals}
        else:
            feed_dict = {self.phs[key]: feed_vals[key] for key in feed_vals}
        return feed_dict
