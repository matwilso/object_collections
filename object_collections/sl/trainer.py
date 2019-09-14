import warnings
warnings.filterwarnings('ignore')
import itertools
import os
import copy

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from object_collections.rl.data import Dataset
from object_collections.sl.viz import plot, image_tile_summary
from tensorflow_probability import distributions as tfd
from .aux_losses import mdn_loss, mdn_metrics
import object_collections.rl.util as rl_util
from object_collections.rl.encoders import DYN, VAE
from .building_blocks import MDN_Head
from .util import multi_head_loss

def numvars(): print("\nNUM VARIABLES", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]), "\n")
def subdict(dict, keys): return {key: dict[key] for key in keys}
def prefix_vals(name, vals): return {name.lower() + '_' + key: vals[key] for key in vals} # add a prefix to dictionary keys
def approx_kl(dist1, dist2, samples):
    rate = dist1.log_prob(samples) - dist2.log_prob(samples)
    return tf.reduce_mean(rate)
def approx_nlogp(dist, val):
    distortion = -dist.log_prob(val)
    return tf.reduce_mean(distortion)

class Trainer(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.is_training_ph = tf.placeholder(tf.bool, [])
        #self.train_ds = Dataset(self.FLAGS)
        self.filenames = copy.deepcopy(self.FLAGS['filenames'])
        self.filenames_ph = tf.placeholder(tf.string)
        self.shuffle_files = lambda: np.random.shuffle(self.filenames)

        self.action_dist = rl_util.convert_to_dict_dist(self.FLAGS['action_space'].spaces)
        self.obs_vectorizer = rl_util.convert_to_dict_dist(self.FLAGS['observation_space'].spaces)

        self.train_ds = Dataset(self.obs_vectorizer, self.FLAGS, filenames_ph=self.filenames_ph)

        Encoder = VAE if 'vae' in self.FLAGS['cnn_gn'] else DYN
        self.encoder = Encoder(self.train_ds.sas_vals, self.train_ds.sas, self.action_dist, self.obs_vectorizer, self.FLAGS, conv='cnn' in self.FLAGS['cnn_gn'], is_training_ph=self.is_training_ph)

        # Summaries
        self.eval_vals = {}
        self.pred_plot_ph = tf.placeholder(tf.uint8)
        self.plot_summ = tf.summary.image('mdn_contour', self.pred_plot_ph)

        self.train_vals = {'losses': self.encoder.losses, 'train_op': self.encoder.train_op}

        # TODO: make this more general in case we want to map from GN to GN
        if self.FLAGS['cnn_train_match'] or self.FLAGS['dyn_coadapt']:
            if self.FLAGS['cnn_train_match']:
                self.conv_dyn = DYN(self.train_ds.sas_vals, self.train_ds.sas, self.action_dist, self.obs_vectorizer, self.FLAGS, conv=True, name='Conv_DYN', mdn_model=self.encoder.mdn, forward_model=self.encoder.model._forward, inverse_model=self.encoder.model._inverse, prior_model=self.encoder.model._prior)

                if self.FLAGS['dyn_prob']:
                    sv = self.encoder.step_vals['phi_s_prob']
                    csv = self.conv_dyn.step_vals['phi_s_prob']
                    prob_losses = {}
                    loss_op = prob_losses['cnn_to_gn_loss'] = approx_kl(csv['dist'], sv['dist'], csv['samples'])

                    #'prior_s_loss'
                    #for key in ['fwd_loss', 'inv_loss', 'mdn_loss', 'prior_s_loss', 'prior_s_next_loss']:
                    for key in ['prior_s_loss', 'prior_s_next_loss']:
                        prob_losses[key] = self.conv_dyn.vals[key]

                    var_list = list(itertools.chain(*[lay.get_variables() for lay in self.conv_dyn.model._encoder.layers]))
                    master_grads_and_vars, self.grad_summaries = multi_head_loss(self.conv_dyn.trainer, prob_losses, self.FLAGS, var_list=var_list)
                    cnn_train_op = self.conv_dyn.trainer.apply_gradients(master_grads_and_vars, global_step=self.conv_dyn.global_step)
                    self.train_vals['train_op'] = cnn_train_op
                    # TODO: maybe group ops for coadapt
                    #self.train_vals['train_op'] = self.conv_dyn.trainer.minimize(loss_op, var_list=var_list)
                    self.train_vals['losses'].update(prob_losses)

                else:
                    loss_op = tf.losses.mean_squared_error(self.encoder.step_vals['phi_s'], self.conv_dyn.step_vals['phi_s'])
                    self.train_vals['losses']['cnn_to_gn_loss'] = loss_op
                    self.train_vals['train_op'] = self.conv_dyn.trainer.minimize(loss_op, var_list=self.conv_dyn.model.get_variables())

            elif self.FLAGS['dyn_coadapt']:
                self.conv_dyn = encoder(self.train_ds.sas_vals, self.train_ds.sas, self.action_dist, self.obs_vectorizer, self.FLAGS, conv=True, name='Conv_DYN', mdn_model=self.encoder.mdn, forward_model=self.encoder.model._forward, inverse_model=self.encoder.model._inverse, prior_model=self.encoder.model._prior)

                sv = self.encoder.step_vals['phi_s_prob']
                csv = self.conv_dyn.step_vals['phi_s_prob']

                prob_losses = {}
                prob_losses['cnn_to_gn_loss'] = approx_kl(csv['dist'], sv['dist'], csv['samples'])
                prob_losses['gn_to_cnn_loss'] = approx_kl(sv['dist'], csv['dist'], sv['samples'])

                loss_names = ['fwd_loss', 'inv_loss', 'mdn_loss', 'prior_s_loss', 'prior_s_next_loss']
                prob_losses.update(prefix_vals('cnn', subdict(self.conv_dyn.vals, loss_names)))
                prob_losses.update(prefix_vals('gn', subdict(self.encoder.vals, loss_names)))
                self.eval_vals['delta_mdn'] = prob_losses['cnn_mdn_loss'] - prob_losses['gn_mdn_loss']

                # TODO: should do things individually or backprop 
                master_grads_and_vars, self.grad_summaries = multi_head_loss(self.encoder.trainer, prob_losses, self.FLAGS)
                train_op = self.encoder.trainer.apply_gradients(master_grads_and_vars, global_step=self.encoder.global_step)

                self.train_vals['train_op'] = train_op
                self.train_vals['losses'].update(prob_losses)
                loss_op = None
            #else:
            #    self.conv_dyn = Encoder(self.train_ds.sas_vals, self.train_ds.sas, self.action_dist, self.obs_vectorizer, self.FLAGS, conv=True, name='Conv_DYN')
            #    self.train_vals['train_op'] = self.conv_dyn.train_op
            #    loss_op = None
            #    prob_losses = {key: val for key, val in self.conv_dyn.vals.items() if 'loss' in key}

            self.eval_vals.update(self.encoder.eval_vals)
            self.eval_vals.update(prefix_vals('cnn', self.conv_dyn.eval_vals))
            self.eval_vals.update(prob_losses)
            self.conv_plot_summ = tf.summary.image('conv_mdn_contour', self.pred_plot_ph)

            # Summaries
            with tf.name_scope('train'):
                eval_summaries = [self.encoder.eval_vals.pop('summary')]
                eval_summaries += [tf.summary.scalar(key, prob_losses[key]) for key in prob_losses]
                eval_summaries = tf.summary.merge(eval_summaries)
            
            with tf.name_scope('cnn_train'):
                cnn_summaries = [self.conv_dyn.eval_vals.pop('summary')]
                cnn_summaries += [tf.summary.image('image', self.train_ds.sas_vals['s']['image'])]
                cnn_summaries = tf.summary.merge(cnn_summaries)

            summary = tf.summary.merge([eval_summaries, cnn_summaries])
            self.eval_vals['summary'] = summary
        else:
            self.eval_vals.update(self.encoder.losses)
            eval_summaries = [self.encoder.eval_vals.pop('summary')]
            with tf.name_scope('train'):
                eval_summaries += [tf.summary.scalar(key, self.encoder.losses[key]) for key in self.encoder.losses]
            summary = tf.summary.merge(eval_summaries + self.encoder.grad_summaries)
            self.eval_vals['summary'] = summary
            self.eval_vals.update(self.encoder.eval_vals)


        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10, keep_checkpoint_every_n_hours=0.5)

        # Load pre-trained variables from a path maybe.  
        if self.FLAGS['load_path'] != '':
            def get_vars_to_load(path):
                load_names = [name for name,_ in tf.contrib.framework.list_variables(path)]
                vars = [var for var in tf.global_variables() if var.name[:-2] in load_names and 'Adam' not in var.name]
                return vars
            path = self.FLAGS['load_path']
            loader = tf.train.Saver(var_list=get_vars_to_load(path))
            loader.restore(self.sess, path)
            print()
            print('Loaded trained variables from', path)
            print()


        if self.FLAGS['shuffle_files']:
            self.shuffle_files()
        self.train_writer = tf.summary.FileWriter(self.FLAGS['log_path'] + '/train', self.sess.graph)
        numvars()


    def run(self):
        self.sess.run(self.train_ds.iterator.initializer, {self.filenames_ph: self.filenames})
        losses = {key: [] for key in self.encoder.losses}
        mean_loss = 99
        #prev_arr = 0
        nepoch = 0
        for i in itertools.count(start=1):
            try:
                if i % self.FLAGS['eval_n'] == 0:
                    #tv, ev = self.sess.run([self.train_vals, self.eval_vals], {self.is_training_ph: True})
                    if self.FLAGS['inference']:
                        ev, phis = self.sess.run([self.eval_vals, self.encoder.step_vals], {self.is_training_ph: True})
                    else:
                        tv, ev, phis = self.sess.run([self.train_vals, self.eval_vals, self.encoder.step_vals], {self.is_training_ph: True})
                        [losses[key].append(tv['losses'][key]) for key in self.encoder.losses]

                    self.train_writer.add_summary(ev['summary'], global_step=i)
                    if 'vae' not in self.FLAGS['cnn_gn']:
                        self.plot_contours(ev, itr=i)

                    mean_loss = {key: np.mean(losses[key]) for key in losses}
                    losses = {key: [] for key in self.encoder.losses}
                    print('i = {}, loss: {}'.format(i, mean_loss))

                    self.saver.save(self.sess, self.FLAGS['save_path'], global_step=i)
                else:
                    tv = self.sess.run(self.train_vals, {self.is_training_ph: True})
                    [losses[key].append(tv['losses'][key]) for key in self.encoder.losses]


                if i >= self.FLAGS['total_n']:
                    return mean_loss
            except tf.errors.OutOfRangeError as e:
                print('Epoch done')
                nepoch += 1
                if nepoch >= self.FLAGS['num_epochs']:
                    return mean_loss
                else:
                    if self.FLAGS['shuffle_files']:
                        self.shuffle_files()
                    self.sess.run(self.train_ds.iterator.initializer, {self.filenames_ph: self.filenames})


    def plot_contours(self, ev, itr):
        plots = []
        pre = 's_mdn_'
        for i in range(3):
            one = plot('contour', {'state': ev[pre+'state']['array'][i], 'X':ev[pre+'X'], 'Y':ev[pre+'Y'], 'Z':ev[pre+'Z'][:,:,i]}, self.FLAGS, itr=itr, return_data=True)
            plots.append(one)
        plots = np.stack(plots)
        plot_summary = self.sess.run(self.plot_summ, {self.pred_plot_ph: plots})
        self.train_writer.add_summary(plot_summary, global_step=itr)

        if 'cnn' in self.FLAGS['cnn_gn'] and (self.FLAGS['cnn_train_match'] or self.FLAGS['dyn_coadapt']):
            plots = []
            pre = 'cnn_s_mdn_'
            for i in range(3):
                one = plot('contour', {'state': ev[pre+'state']['array'][i], 'X':ev[pre+'X'], 'Y':ev[pre+'Y'], 'Z':ev[pre+'Z'][:,:,i]}, self.FLAGS, itr=itr, return_data=True)
                plots.append(one)
            plot_summary = self.sess.run(self.conv_plot_summ, {self.pred_plot_ph: plots})
            self.train_writer.add_summary(plot_summary, global_step=itr)

    def evaluate(self, filenames):
        pre = 's_mdn_'
        self.sess.run(self.train_ds.iterator.initializer, {self.filenames_ph: filenames})

        if self.FLAGS['bs'] == 1:
            for itr in itertools.count():
                tv, ev = self.sess.run([self.train_vals, self.eval_vals], {self.is_training_ph: True})
                X, Y, Z = ev[pre+'X'], ev[pre+'Y'], np.rot90(ev[pre+'Z'][...,0], k=3)
                fig, (ax1, ax2) = plt.subplots(2,1)
                ax1.contour(X, Y, Z, alpha=0.5)
                ax1.set_aspect('equal')
                ax2.imshow(ev[pre+'state']['image'][0])
                ax2.set_aspect('equal')
                plt.savefig('test{}.png'.format(itr))
                #plt.show()
        else:
            tv, ev = self.sess.run([self.train_vals, self.eval_vals], {self.is_training_ph: self.FLAGS['is_training']})
            print('Success')
            for itr in itertools.count():
                X, Y, Z = ev[pre+'X'], ev[pre+'Y'], np.rot90(ev[pre+'Z'][...,itr], k=3)
                fig, (ax1, ax2) = plt.subplots(2,1)
                ax1.contour(X, Y, Z, alpha=0.5)
                ax1.set_aspect('equal')
                ax2.imshow(ev[pre+'state']['image'][itr])
                ax2.set_aspect('equal')
                plt.savefig('reals/bnlive{}.png'.format(itr))
                #plt.show()


