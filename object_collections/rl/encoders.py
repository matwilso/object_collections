import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from anyrl.algos.advantages import GAE
from anyrl.models import TFActorCritic

import sonnet as snt
import object_collections.rl.util as rl_util
from graph_nets import utils_tf
from object_collections.envs import discretize, map_continuous, unmap_continuous
from object_collections.envs.utils.table import TABLES
from object_collections.sl.aux_losses import (mask_state, mdn_loss, mdn_metrics, vae_loss_and_metrics)
                                      
from object_collections.sl.building_blocks import (GraphFormer, MDN_Head, Transformer)
from object_collections.rl.models import DYN_Model, ConvVAE_Model, GNVAE_Model
from tensorflow_probability import distributions as tfd
from object_collections.sl.util import multi_head_loss
from object_collections.tf_util import get_session

def subdict(dict, keys): return {key: dict[key] for key in keys}
def prefix_vals(name, vals): return {name.lower() + '_' + key: vals[key] for key in vals} # add a prefix to dictionary keys

class DYN(object):
    def __init__(self, sas_vals, sas_phs, action_dist, obs_vectorizer, FLAGS, conv=False, goal_model=None, mdn_model=None, forward_model=None, inverse_model=None, prior_model=None, compute_grad=True, is_training_ph=None, name='DYN'):
        self.name = name
        with tf.variable_scope(self.name):
            self.FLAGS = FLAGS
            self.action_dist = action_dist
            self.obs_vectorizer = obs_vectorizer
            # phs and tf return vals
            self.step_phs = dict(sas_phs)
            self.step_vals = {}
            self.phs = dict(sas_phs)
            self.vals = {}
            self.eval_vals = {}
            self.is_training_ph = is_training_ph

            self.model = DYN_Model(action_dist, self.FLAGS, conv=conv, inverse=inverse_model, forward=forward_model, prior=prior_model, is_training_ph=is_training_ph)
            self.model_inputs = {}
            self.mdn = mdn_model or MDN_Head(self.FLAGS, conv=conv, activ=self.FLAGS['ACTIVS'][self.FLAGS['cnn_activ']])

            self.model_inputs = sas_vals
            self.step_vals.update(self.model(self.model_inputs)) #{'phi_s', 'phi_s_next', 'phi_s_next_pred', 'a_pred', + prob stuff if dyn_prob}

            # MDN stuff
            phi_s, phi_s_next = self.step_vals['phi_s'], self.step_vals['phi_s_next']
            if self.FLAGS['abs_noise']:
                phi_s += tf.random_normal(tf.shape(phi_s), stddev=self.FLAGS['phi_noise'])
                phi_s_next += tf.random_normal(tf.shape(phi_s), stddev=self.FLAGS['phi_noise'])
            else:
                batch_mean, batch_std = tf.nn.moments(phi_s, axes=0)
                phi_s += tf.random_normal(tf.shape(phi_s), stddev=self.FLAGS['phi_noise']*batch_std)
                phi_s_next += tf.random_normal(tf.shape(phi_s), stddev=self.FLAGS['phi_noise']*batch_std)

            self.s_mdn_vals = self.mdn(phi_s)
            self.s_next_mdn_vals = self.mdn(phi_s_next)

            s_mdn_eval_vals = mdn_metrics(self.model_inputs['s'], self.s_mdn_vals, self.FLAGS)
            self.step_vals.update(prefix_vals('s_mdn', subdict(self.s_mdn_vals, ['locs', 'logits', 'scales', 'flattened'])))
            self.step_vals.update(prefix_vals('s_next_mdn', subdict(self.s_next_mdn_vals, ['locs', 'logits', 'scales', 'flattened'])))

            # Loss
            self.eval_vals['summary'] = s_mdn_eval_vals.pop('summary')
            self.eval_vals.update(prefix_vals('s_mdn', s_mdn_eval_vals))

            square_error = tf.square(self.step_vals['phi_s_next_pred'] - self.step_vals['phi_s_next'])
            self.goal_model = goal_model or self.model

            if self.FLAGS['goal_conditioned']:
                g_state = self.model_inputs['g']
                self.step_vals['phi_g'] = self.goal_model({'s': g_state})['phi_s']
                self.g_mdn_vals = self.mdn(self.step_vals['phi_g'])
                self.step_vals.update(prefix_vals('g_mdn', subdict(self.g_mdn_vals, ['locs', 'logits', 'scales', 'flattened'])))

            if self.FLAGS['dyn_prob']:
                # Probabilistic loss
                def approx_kl(dist1, dist2, samples):
                    rate = dist1.log_prob(samples) - dist2.log_prob(samples)
                    return tf.reduce_mean(rate)
                
                def approx_nlogp(dist, val):
                    distortion = -dist.log_prob(val)
                    return tf.reduce_mean(distortion)

                sv = self.step_vals['phi_s_prob']
                svn = self.step_vals['phi_s_next_prob']
                # Embedding to match prior
                if self.FLAGS['enforce_prior']:
                    self.vals['prior_s_loss'] = approx_kl(sv['dist'], sv['prior'], sv['samples'])
                    self.vals['prior_s_next_loss'] = approx_kl(svn['dist'], svn['prior'], svn['samples'])

                # Forward 
                psvn = self.step_vals['phi_s_next_pred_prob']
                self.vals['fwd_loss'] = approx_kl(psvn['dist'], svn['dist'], psvn['samples'])
                self.vals['fwd_loss'] *= self.FLAGS['fwd_coeff']

                # Inverse
                a = self.step_vals['a_pred_prob']
                # TODO: try this other loss
                # TODO: try adding noise also
                self.vals['inv_loss'] = -tf.reduce_mean(self.action_dist.log_prob(self.step_vals['a_pred'], self.step_phs['a']))
                #self.vals['inv_loss'] = approx_nlogp(a['dist'], self.model_inputs['a'])

                # MDN
                self.vals['mdn_loss'] = mdn_loss(self.model_inputs['s'], self.s_mdn_vals, FLAGS)

                # Total
                self.vals['loss'] = sum([self.vals[key] for key in self.vals if 'loss' in key])
            else:
                # Standard loss
                self.vals['fwd_loss'] = 0.5 * tf.reduce_mean(square_error)
                self.vals['inv_loss'] = -tf.reduce_mean(self.action_dist.log_prob(self.step_vals['a_pred'], self.step_phs['a']))
                self.vals['mdn_loss'] = mdn_loss(self.model_inputs['s'], self.s_mdn_vals, FLAGS)

                # TODO: this should be fixed to not do a reduce mean over the rep_size. For now, this fixes loss scale issues
                self.vals['fwd_loss'] *= self.FLAGS['dyn_rep_size']
                self.vals['fwd_loss'] *= self.FLAGS['fwd_coeff']
                self.vals['loss'] = (self.vals['inv_loss'] + self.vals['fwd_loss'] + self.vals['mdn_loss'])

            if compute_grad:
                self.vals['mdn_loss'] *= self.FLAGS['mdn_weight']
                self.vals['fwd_loss'] *= self.FLAGS['dyn_weight']
                self.vals['inv_loss'] *= self.FLAGS['dyn_weight']

                # TODO: add support for annealing learning rate
                self.global_step = tf.get_variable('dyn_global_step', initializer=tf.constant(1, dtype=tf.int32), trainable=False)
                self.trainer = tf.train.AdamOptimizer(learning_rate=self.FLAGS['dyn_lr'], beta1=self.FLAGS['beta1']) 
                self.losses = {key: self.vals[key] for key in self.vals if 'loss' in key and key != 'loss'}
                master_grads_and_vars, self.grad_summaries = multi_head_loss(self.trainer, self.losses, self.FLAGS)
                self.train_op = self.trainer.apply_gradients(master_grads_and_vars, global_step=self.global_step)


    def step(self, sas):
        """Apply DYN to a sas dict to yield reward_signal and original values for clipping in dict"""
        feed_dict = {self.step_phs[key]: sas[key] for key in sas}
        sess = get_session()
        subkeys = [key for key in self.step_vals if 'prob' not in key]
        if self.is_training_ph is not None:
            feed_dict.update({self.is_training_ph: self.FLAGS['is_training']})
        out = sess.run(subdict(self.step_vals, subkeys), feed_dict)
        return out

    def embed_state(self, state):
        feed_dict = {self.step_phs['s']: self.obs_vectorizer.to_vecs(state)}
        sess = get_session()
        out = sess.run(self.step_vals['phi_s'], feed_dict)
        return out

    def off_policy_feed_dict(self, sarsd):
        feed_vals = sarsd
        feed_dict = {self.phs[key]: feed_vals[key] for key in self.phs}
        return feed_dict

    def feed_dict(self, rollouts, batch, sas, orig_vals):
        feed_vals = {key: rl_util.select_from_batch(sas[key], batch) for key in sas}
        feed_dict = {self.phs[key]: feed_vals[key] for key in self.phs}
        return feed_dict
    

class VAE(object):
    def __init__(self, sas_vals, sas_phs, action_dist, obs_vectorizer, FLAGS, compute_grad=True, is_training_ph=None, conv=False, name='VAE', **kwargs):
        self.name = name
        with tf.variable_scope(self.name):
            self.FLAGS = FLAGS
            self.conv = conv
            self.action_dist = action_dist
            self.obs_vectorizer = obs_vectorizer
            # phs and tf return vals
            self.step_phs = dict(sas_phs)
            self.step_vals = {}
            self.phs = dict(sas_phs)
            self.vals = {}
            self.eval_vals = {}
            self.is_training_ph = is_training_ph
            
            if self.conv:
                self.model = ConvVAE_Model(self.FLAGS, is_training_ph=self.is_training_ph)
            else:
                self.model = GNVAE_Model(self.FLAGS, is_training_ph=self.is_training_ph)

            self.model_inputs = sas_vals
            model_out = self.model(self.model_inputs) # 'prior', 'q', 'q_mean', 'q_sample', 'p_hat'
            #self.step_vals.update(model_out)
            self.step_vals['phi_s'] = model_out['q_mean']
            next_model_out = self.model({'s': self.model_inputs['s_next']}) # 'prior', 'q', 'q_mean', 'q_sample', 'p_hat'
            self.step_vals['phi_s_next'] = next_model_out['q_mean']

            # VAE stuff
            vae_loss, s_vae_vals = vae_loss_and_metrics(self.model_inputs['s'], model_out, self.FLAGS)

            if self.FLAGS['goal_conditioned']:
                g_state = self.model_inputs['g']
                goal_model_out = self.model({'s': g_state})
                #self.step_vals.update(prefix_vals('goal', goal_model_out))
                self.step_vals['phi_g'] = goal_model_out['q_mean']

            # Loss
            self.eval_vals['summary'] = s_vae_vals.pop('summary')
            self.eval_vals.update(prefix_vals('s_vae', s_vae_vals))
            self.eval_vals['recon'] = tf.sigmoid(model_out['logits'])

            self.vals['rate_loss'] = s_vae_vals['rate']
            self.vals['distortion_loss'] = s_vae_vals['distortion']
            self.vals['loss'] = vae_loss

            self.losses = self.vals

            if compute_grad:
                self.global_step = tf.get_variable('dyn_global_step', initializer=tf.constant(1, dtype=tf.int32), trainable=False)
                self.trainer = tf.train.AdamOptimizer(learning_rate=self.FLAGS['dyn_lr'], beta1=self.FLAGS['beta1']) 
                self.train_op = self.trainer.minimize(vae_loss)
            self.grad_summaries = []

    def step(self, sas):
        feed_dict = {self.step_phs[key]: sas[key] for key in sas}
        sess = get_session()
        if self.is_training_ph is not None:
            feed_dict.update({self.is_training_ph: self.FLAGS['is_training']})
        #run_vals = {key: val for key, val in self.step_vals.items() if not isinstance(val, tfp.distributions.Distribution)}
        run_vals = self.step_vals
        out = sess.run(run_vals, feed_dict)
        return out

    def off_policy_feed_dict(self, sarsd):
        feed_vals = sarsd
        feed_dict = {self.phs[key]: feed_vals[key] for key in self.phs}
        return feed_dict

    def feed_dict(self, rollouts, batch, sas, orig_vals):
        feed_vals = {key: rl_util.select_from_batch(sas[key], batch) for key in sas}
        feed_dict = {self.phs[key]: feed_vals[key] for key in self.phs}
        return feed_dict
    
