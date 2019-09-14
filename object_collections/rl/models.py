import numpy as np
import tensorflow as tf
from anyrl.algos.advantages import GAE
from anyrl.algos.util import symmetric_clipped_value_loss
from anyrl.models import FeedforwardAC
from object_collections.sl.constant import IMAGE_SHAPE

import sonnet as snt
from graph_nets import utils_tf
from object_collections.sl.building_blocks import MDN_Head, GraphFormer, Transformer, Module, ConvEmbed, MixturePrior, EncoderConv, DecoderConvT, MLPEmbed
from tensorflow_probability import distributions as tfd
from object_collections.rl.util import flatten_mdn

# TODO: add option to not process goal by this model, but to use a different model
# TODO: combine as much logic as possible here
# inverse of tf.nn.softplus
softplus_inverse = lambda x: tf.log(tf.expm1(x))

class AC(object):
    def __init__(self):
        raise NotImplementedError

    @property
    def stateful(self):
        return False

    def start_state(self, batch_size):
        return None

    def step(self, observations, states):
        raise NotImplementedError

    def batch_outputs(self):
        return self.actor_out, self.critic_out

class GNVAE_Model(Module):
    def __init__(self, FLAGS, init=None, activ=None, name='GNVAE_Model', is_training_ph=None):
        super().__init__(FLAGS=FLAGS, init=init, activ=activ, name=name)
        with self._enter_variable_scope():
            self._encoder = GraphFormer(FLAGS)
            self._decoder_conv = DecoderConvT(FLAGS)
            #self._decoder_conv = snt.BatchApply(DecoderConvT(FLAGS))
            self._loc = snt.Linear(FLAGS['vae_z_size'], name='fc_mu', initializers=self.init)
            self._log_scale = snt.Linear(FLAGS['vae_z_size'], name='fc_log_var', initializers=self.init)
            self._prior = MixturePrior(FLAGS['vae_k'], FLAGS['vae_z_size'], FLAGS)
            self._mdn_head = MDN_Head(FLAGS)

    def _build(self, inputs):
        prior = self._prior()
        enc_out = self._encoder(inputs['s'])
        loc = self._loc(enc_out)
        log_scale = self._log_scale(enc_out)
        scale = tf.nn.softplus(log_scale + softplus_inverse(1.0)) # idk what this is for. maybe ensuring center around 1.0

        q = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale, name='code') # approximate posterior
        q_sample = q.sample(self.FLAGS['num_vae_samples'])  # approximate posterior sample
        logits = self._decoder_conv(loc)
        #logits = self._decoder_conv(q_sample)
        phat = tfd.Independent(tfd.Bernoulli(logits=logits), reinterpreted_batch_ndims=len(IMAGE_SHAPE), name="image") 
        return dict(prior=prior, q=q, q_mean=loc, q_sample=q_sample, phat=phat, phi_s=loc, logits=logits)

class ConvVAE_Model(Module):
    def __init__(self, FLAGS, init=None, activ=None, name='ConvVAE_Model', is_training_ph=None):
        super().__init__(FLAGS=FLAGS, init=init, activ=activ, name=name)
        with self._enter_variable_scope():
            self._encoder_conv = EncoderConv(FLAGS)
            self._decoder_conv = DecoderConvT(FLAGS)
            #self._decoder_conv = snt.BatchApply(DecoderConvT(FLAGS))
            self._loc = snt.Linear(FLAGS['vae_z_size'], name='fc_mu', initializers=self.init)
            self._log_scale = snt.Linear(FLAGS['vae_z_size'], name='fc_log_var', initializers=self.init)
            self._prior = MixturePrior(FLAGS['vae_k'], FLAGS['vae_z_size'], FLAGS)
            self._mdn_head = MDN_Head(FLAGS)

    def _build(self, inputs):
        prior = self._prior()
        enc_out = self._encoder_conv(inputs['s']['image'], conv_flatten=self.FLAGS['conv_flatten'])
        loc = self._loc(enc_out)
        log_scale = self._log_scale(enc_out)
        scale = tf.nn.softplus(log_scale + softplus_inverse(1.0)) # idk what this is for. maybe ensuring center around 1.0

        q = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale, name='code') # approximate posterior
        q_sample = q.sample(self.FLAGS['num_vae_samples'])  # approximate posterior sample
        logits = self._decoder_conv(loc)
        #logits = self._decoder_conv(q_sample)
        phat = tfd.Independent(tfd.Bernoulli(logits=logits), reinterpreted_batch_ndims=len(IMAGE_SHAPE), name="image") 
        return dict(prior=prior, q=q, q_mean=loc, q_sample=q_sample, phat=phat, phi_s=loc, logits=logits)

class DYN_Model(Module):
    """""" 
    def __init__(self, action_dist, FLAGS, activ=None, init=None, conv=False, forward=None, inverse=None, prior=None, is_training_ph=None, name="DYN_Model"):
        self.conv = conv
        if activ is None:
            if self.conv:
                activ = FLAGS['ACTIVS'][FLAGS['cnn_activ']]
            else:
                activ = FLAGS['ACTIVS'][FLAGS['gn_activ']]

        super().__init__(FLAGS=FLAGS, init=init, activ=activ, name=name)

        self._prior = None
        # TODO: remove this. it was just a temporary hack
        self.hidden_size = self.FLAGS['cnn_hidden_size'] if conv else self.FLAGS['tf_hidden_size']

        with self._enter_variable_scope():
            if self.conv:
                embed = ConvEmbed(self.FLAGS, use_gformer='cnn_gn' in self.FLAGS['cnn_gn'], conv_flatten=self.FLAGS['conv_flatten'], is_training_ph=is_training_ph, activ=self.FLAGS['ACTIVS'][self.FLAGS['cnn_activ']])
            else:
                if 'mlp' in self.FLAGS['cnn_gn']:
                    embed = MLPEmbed(self.FLAGS, activ=self.FLAGS['ACTIVS'][self.FLAGS['gn_activ']])
                else:
                    embed = GraphFormer(self.FLAGS, activ=self.FLAGS['ACTIVS'][self.FLAGS['gn_activ']])

            self.action_shape = np.product(action_dist.out_shape)

            if self.FLAGS['dyn_prob']:
                self._encoder = snt.Sequential([
                    embed,
                    snt.Linear(2*self.FLAGS['dyn_rep_size'], initializers=self.FLAGS['init']),
                ], 'Encoder')
                if prior is None:
                    self._prior = MixturePrior(k=self.FLAGS['vae_k'], size=self.FLAGS['dyn_rep_size'], FLAGS=self.FLAGS)
                else:
                    self._prior = prior
                if forward is None:
                    self._forward = snt.BatchApply(snt.nets.MLP([self.hidden_size, 2*self.FLAGS['dyn_rep_size']], activation=self.activ, initializers=self.init, name='fwd'))
                else:
                    self._forward = forward

                if inverse is None:
                    self._inverse = snt.nets.MLP([self.hidden_size, 2*self.action_shape], activation=self.activ, initializers=self.init, name='inv')
                    #self._inverse = snt.BatchApply(snt.nets.MLP([self.hidden_size, 2*self.action_shape], activation=self.activ, initializers=self.init, name='inv'))
                else:
                    self._inverse = inverse

            else:
                self._encoder = snt.Sequential([
                    embed,
                    snt.Linear(self.FLAGS['dyn_rep_size'], initializers=self.FLAGS['init']),
                ], 'Encoder')

                if forward is None:
                    self._forward = snt.nets.MLP([self.hidden_size, self.FLAGS['dyn_rep_size']], activation=self.activ, initializers=self.init, name='fwd')
                else:
                    self._forward = forward

                if inverse is None:
                    self._inverse = snt.nets.MLP([self.hidden_size, 2*self.action_shape], activation=self.activ, initializers=self.init, name='inv')
                else:
                    self._inverse = inverse

    def _build(self, sas):
        vals = {}
        #from object_collections.sl.models import ConvVAE
        #cc = ConvVAE(self.FLAGS)
        #cc(sas['s'])
        if self.FLAGS['dyn_prob']:
            prior = self._prior()
            if 's' in sas:
                phi_s = self._encoder(sas['s'])
                loc, log_scale = phi_s[...,:self.FLAGS['dyn_rep_size']], phi_s[...,self.FLAGS['dyn_rep_size']:]
                scale = tf.nn.softplus(log_scale + softplus_inverse(1.0))
                dist = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale, name='dyn_phi') # approximate posterior
                vals['phi_s'] = dist.mean()
                vals['phi_s_prob'] = dict(prior=prior, dist=dist, samples=dist.sample(self.FLAGS['num_vae_samples']))
            if 's_next' in sas:
                phi_s_next = self._encoder(sas['s_next'])
                loc, log_scale = phi_s_next[...,:self.FLAGS['dyn_rep_size']], phi_s_next[...,self.FLAGS['dyn_rep_size']:]
                scale = tf.nn.softplus(log_scale + softplus_inverse(1.0))
                dist = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale, name='dyn_phi_next') # approximate posterior
                vals['phi_s_next'] = dist.mean()
                vals['phi_s_next_prob'] = dict(prior=prior, dist=dist, samples=dist.sample(self.FLAGS['num_vae_samples']))
            if 'a' in sas:
                # Next prediction
                tile_up = lambda x: tf.tile(x[None], [self.FLAGS['num_vae_samples'], 1, 1])
                next_pred = self._forward(tf.concat([tile_up(sas['a']), vals['phi_s_prob']['samples']], axis=-1))
                loc, log_scale = next_pred[...,:self.FLAGS['dyn_rep_size']], next_pred[...,self.FLAGS['dyn_rep_size']:]
                scale = tf.nn.softplus(log_scale + softplus_inverse(1.0))
                dist = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale, name='dyn_phi_next_pred')
                vals['phi_s_next_pred'] = dist.mean()
                # TODO: do we need a sample from this?
                vals['phi_s_next_pred_prob'] = dict(dist=dist, samples=dist.sample(self.FLAGS['num_vae_samples']))

                # Action prediction
                a_pred = self._inverse(tf.concat([vals['phi_s'], vals['phi_s_next']], axis=-1))
                #a_pred = self._inverse(tf.concat([tile_up(vals['phi_s']), vals['phi_s_next_prob']['samples']], axis=-1))
                assert self.FLAGS['discrete'] == False
                loc, log_scale = a_pred[...,:self.action_shape], a_pred[...,self.action_shape:]
                scale = tf.nn.softplus(log_scale + softplus_inverse(1.0))
                dist = tfd.MultivariateNormalDiag(loc, scale)
                #vals['a_pred'] = dist.mean()
                vals['a_pred'] = a_pred
                vals['a_pred_prob'] = dict(dist=dist)
            return vals
        else:
            if 's' in sas:
                vals['phi_s'] = self._encoder(sas['s'])
            if 's_next' in sas:
                vals['phi_s_next'] = self._encoder(sas['s_next'])
            if 'a' in sas:
                vals['phi_s_next_pred'] = self._forward(tf.concat([sas['a'], vals['phi_s']], axis=-1))
                vals['a_pred'] = self._inverse(tf.concat([vals['phi_s'], vals['phi_s_next']], axis=-1))
            return vals

class SACAC(AC):
    """ """
    def __init__(self, sess, sas_vals, sas_phs, embed_phs, action_dist, obs_vectorizer, FLAGS, dyn=None, value_dyn=None, act_limit=1.0, is_training_ph=None):
        """ """
        self.FLAGS = FLAGS
        self.sess = sess
        self.action_dist = action_dist
        self.obs_vectorizer = obs_vectorizer
        self.sas_phs = sas_phs
        self.embed_phs = embed_phs
        self.act_limit = act_limit
        self.is_training_ph = is_training_ph
        assert is_training_ph is not None, 'gotta set this'

        self.main_model = SAC_Module(action_dist.out_shape, self.FLAGS, dyn=dyn, value_dyn=value_dyn)
        self.target_model = SAC_Module(action_dist.out_shape, self.FLAGS, dyn=dyn, value_dyn=value_dyn, name='Target_SAC_Module', mode='target')

        # Main model 
        main_inputs = {'s': sas_vals['s'], 'a': sas_vals['a']}
        if FLAGS['goal_conditioned']:
            main_inputs.update({'g': sas_vals['g']})
        self.main_out = self.main_model(main_inputs)

        # Target model
        target_inputs = {'s': sas_vals['s_next']}
        if FLAGS['goal_conditioned']:
            target_inputs.update({'g': sas_vals['g']})
        self.target_out = self.target_model(target_inputs)

        self.step_out = self.main_out

        if self.FLAGS['use_embed']:
            main_inputs = embed_phs
            self.main_out = self.main_model(main_inputs, mode='not_step')
            if self.FLAGS['aac']:
                target_inputs = {'phi_s_vf': embed_phs['phi_s_next_vf'], 'phi_g_vf': embed_phs['phi_g_vf']}
            else:
                target_inputs = {'phi_s_pi': embed_phs['phi_s_next_pi'], 'phi_g_pi': embed_phs['phi_g_pi']}
            self.target_out = self.target_model(target_inputs, mode='not_step')

    def step(self, observations, states, mode='explore'):
        if mode == 'random': 
            actions = np.random.uniform(-1, 1, (len(observations),) + self.action_dist.out_shape)
        else:
            if mode == 'exploit':
                feed_dict = {self.sas_phs['s']: self.obs_vectorizer.to_vecs(observations), self.is_training_ph: self.FLAGS['is_training']}
                actions = self.sess.run(self.step_out['mu'], feed_dict)
            else:
                feed_dict = {self.sas_phs['s']: self.obs_vectorizer.to_vecs(observations), self.is_training_ph: self.FLAGS['is_training']}
                actions = self.sess.run(self.step_out['pi'], feed_dict)

        return {
            'actions': actions,
            'states': None,
        }

    def batch_outputs(self):
        return self.main_out, self.target_out

EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -20

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)

def apply_squashing_func(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi


class SAC_Module(Module):
    def __init__(self, out_shape, FLAGS, activ=None, init=None, name="SAC_Module", dyn=None, value_dyn=None, mode='main'):
        super().__init__(FLAGS=FLAGS, init=init, activ=activ, name=name)
        self.dyn = dyn
        self.value_dyn = value_dyn
        self.mode = mode
        self.out_shape = out_shape

        with self._enter_variable_scope():
            if self.dyn is None:
                PolicyBase = GraphFormer
            else:
                def PolicyBase(*args, **kwargs):
                    def _thunk(x):
                        if 'vae' in self.FLAGS['cnn_gn']:
                            phi_s = self.dyn.model({'s': x})['q_mean']
                        else:
                            phi_s = self.dyn.model({'s': x})['phi_s']
                        return phi_s

                    return _thunk

                def ValueBase(*args, **kwargs):
                    def _thunk(x):
                        phi_s = self.value_dyn.model({'s': x})['phi_s']
                        return phi_s
                    return _thunk

            if self.FLAGS['aac']:
                assert self.value_dyn is not None
                self._value_base = ValueBase(self.FLAGS, mode=mode)
                self._policy_base = PolicyBase(self.FLAGS, mode=mode)
            else:
                self._policy_base = PolicyBase(self.FLAGS, mode=mode)
                self._value_base = self._policy_base

            if mode == 'main':
                self._pi_net = snt.nets.MLP([self.FLAGS['tf_hidden_size']]*2 + [out_shape[0]*2], activation=self.FLAGS['activ'], name='pi_net')
                self._q1_net = snt.nets.MLP([self.FLAGS['tf_hidden_size']]*2 + [1], activation=self.FLAGS['activ'], name='q1_net')
                self._q2_net = snt.nets.MLP([self.FLAGS['tf_hidden_size']]*2 + [1], activation=self.FLAGS['activ'], name='q2_net')
            self._v_net = snt.nets.MLP([self.FLAGS['tf_hidden_size']]*2 + [1], activation=self.FLAGS['activ'], name='v_net')
           
    def _build(self, inputs, mode='step', act_limit=1.0):
        out = {}
        if mode != 'step' and self.FLAGS['use_embed']:
            if self.mode == 'main' or not self.FLAGS['aac']:
                if self.FLAGS['aac'] and self.FLAGS['value_goal']:
                    # use goal embedding from the value network (usually raw state in gn)
                    policy_base = tf.concat([inputs['phi_s_pi'], inputs['phi_g_vf']], axis=-1)
                else:
                    policy_base = tf.concat([inputs['phi_s_pi'], inputs['phi_g_pi']], axis=-1)
            if self.FLAGS['aac']:
                value_base = tf.concat([inputs['phi_s_vf'], inputs['phi_g_vf']], axis=-1)
            else:
                value_base = policy_base
        else:
            #if not self.FLAGS['use_embed'] and self.FLAGS['aac']:
            #    assert False, 'not supported'
            policy_base = self._policy_base(inputs['s'])
            if self.FLAGS['goal_conditioned']:
                def embed_goal(base_fn, inputs): 
                    state = {'graph': inputs['g']['graph'], 'array': inputs['g']['array']}
                    if self.FLAGS['use_image']:
                        state['image'] = inputs['g']['image']
                    return base_fn(state)

                if self.FLAGS['aac'] and self.FLAGS['value_goal']:
                    goal_base = tf.stop_gradient(embed_goal(self._value_base, inputs)) # stop_gradient just to save compute
                else:
                    goal_base = tf.stop_gradient(embed_goal(self._policy_base, inputs)) # stop_gradient just to save compute

                policy_base = tf.concat([policy_base, goal_base], axis=-1)
            value_base = policy_base

        if self.mode == 'main':
            # Compute Gaussian policy stuff
            with tf.variable_scope('pi'):
                pi_raw = self._pi_net(policy_base)
                mu, log_std = pi_raw[...,:self.out_shape[0]], pi_raw[...,self.out_shape[0]:]
                # Make sure log is in reasonable region (see spinningup repo for more detailed explanation)
                log_std = tf.tanh(log_std)
                log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
                std = tf.exp(log_std)

                # Reparameterization trick so we can backprop.  Sample from independent.
                #pi = mu + tf.random_normal(tf.shape(mu)) * std
                if self.FLAGS['inference']:
                    std *= 0.5
                pi = mu + tfd.MultivariateNormalDiag(loc=tf.zeros_like(mu), scale_diag=tf.ones_like(mu)).sample() * std
                logp_pi = tfd.MultivariateNormalDiag(loc=mu, scale_diag=std).log_prob(pi)
                mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

            mu *= act_limit
            pi *= act_limit

            out['q1'] = self._q1_net(tf.concat([value_base, inputs['a']], axis=-1))[...,0]
            out['q2'] = self._q2_net(tf.concat([value_base, inputs['a']], axis=-1))[...,0]
            out['q1_pi'] = self._q1_net(tf.concat([value_base, pi], axis=-1))[...,0]
            out['q2_pi'] = self._q2_net(tf.concat([value_base, pi], axis=-1))[...,0]
            out['v'] = self._v_net(value_base)[...,0]
            out['pi'] = pi
            out['mu'] = mu
            out['logp_pi'] = logp_pi
        elif self.mode == 'target':
            out['v'] = self._v_net(value_base)[...,0]
        return out
