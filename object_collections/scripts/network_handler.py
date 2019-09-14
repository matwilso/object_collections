import itertools
import os
import time
from collections import defaultdict
from multiprocessing import Process, Queue
from threading import Thread
import queue
import scipy
cos_dist = scipy.spatial.distance.cosine

from object_collections.sl.util import convert_nodes_to_graph_tuple
import cloudpickle
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from baselines import logger
from gym.spaces import Tuple

import object_collections.rl.util as rl_util
from object_collections.envs import TABLES
from object_collections.sl.viz import image_tile_summary, plot
from object_collections.rl.agents import Scripted, SAC
from object_collections.rl.encoders import DYN, VAE
from object_collections.rl.mpi_util import RunningMeanVar
from object_collections.tf_util import get_session

from object_collections.rl.data import rollout_to_pickle, rollout_to_tf_record
from object_collections.rl import rollers


def prefix_vals(name, vals): return {name.lower() + '_' + key: vals[key] for key in vals} # add a prefix based on the name (agent or dyn)
def subdict(dict, keys): return {key: dict[key] for key in keys}
def flatten_list_dict(sarsd): return {key: np.concatenate(sarsd[key]) for key in sarsd}

def mean_reward(rollouts): return np.mean([np.mean(r.rewards) for r in rollouts])
def abs_mean_reward(rollouts): return np.mean([np.mean(np.abs(r.rewards)) for r in rollouts])
def mean_return(rollouts): return np.sum([np.sum(r.rewards) for r in rollouts]) / len(rollouts)
def abs_mean_return(rollouts): return np.sum([np.sum(np.abs(r.rewards)) for r in rollouts]) / len(rollouts)

def dump_data(data, path): 
    with open(path, 'wb') as f: 
        cloudpickle.dump(data, f)

numvars = lambda : print("\nNUM VARIABLES", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]), "\n")

class NetworkHandler:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.action_dist = rl_util.convert_to_dict_dist(self.FLAGS['action_space'].spaces)
        self.obs_vectorizer = rl_util.convert_to_dict_dist(self.FLAGS['observation_space'].spaces)
        self.is_training_ph = tf.placeholder(tf.bool, [])

        # TF INIT STUFF
        # TODO: make these passable as cmd line params
        self.global_itr = tf.get_variable('global_itr', initializer=tf.constant(1, dtype=tf.int32), trainable=False)
        self.inc_global = tf.assign_add(self.global_itr, tf.constant(1, dtype=tf.int32))

        # sarsd phs
        in_batch_shape = (None,) + self.obs_vectorizer.out_shape
        self.sarsd_phs = {}
        self.sarsd_phs['s'] = tf.placeholder(tf.float32, shape=in_batch_shape, name='s_ph')
        self.sarsd_phs['a'] = tf.placeholder(tf.float32, (None,) + self.action_dist.out_shape, name='a_ph')  # actions that were taken
        self.sarsd_phs['s_next'] = tf.placeholder(tf.float32, shape=in_batch_shape, name='s_next_ph')
        self.sarsd_phs['r'] = tf.placeholder(tf.float32, shape=(None,), name='r_ph')
        self.sarsd_phs['d'] = tf.placeholder(tf.float32, shape=(None,), name='d_ph')

        self.embed_phs = {key: tf.placeholder(tf.float32, shape=(None, self.FLAGS['embed_shape']), name='{}_ph'.format(key)) for key in self.FLAGS['embeds']}
        for key in ['a', 'r', 'd']: self.embed_phs[key] = self.sarsd_phs[key]

        # Pre-compute these transforms so we don't have to do it all the time
        # sarsd vals
        self.sarsd_vals = rl_util.sarsd_to_vals(self.sarsd_phs, self.obs_vectorizer, self.FLAGS)
        Encoder = VAE if 'vae' in self.FLAGS['cnn_gn'] else DYN
        EName = 'VAE' if 'vae' in self.FLAGS['cnn_gn'] else 'DYN'

        # ALGO SETUP
        if self.FLAGS['goal_dyn'] != '':
            self.goal_model = DYN(self.sarsd_vals, self.sarsd_phs, self.action_dist, self.obs_vectorizer, FLAGS, conv='cnn' in self.FLAGS['goal_dyn'], name='GoalDYN', compute_grad=False, is_training_ph=self.is_training_ph).model
        else:
            self.goal_model = None

        if self.FLAGS['aac']:
            self.value_dyn = Encoder(self.sarsd_vals, self.sarsd_phs, self.action_dist, self.obs_vectorizer, FLAGS, conv=False, name='Value'+EName, is_training_ph=self.is_training_ph, compute_grad=False)
            if self.FLAGS['value_goal']:
                self.goal_model = self.value_dyn.model
        else:
            self.value_dyn = None

        self.dyn = Encoder(self.sarsd_vals, self.sarsd_phs, self.action_dist, self.obs_vectorizer, FLAGS, conv='cnn' in self.FLAGS['cnn_gn'], goal_model=self.goal_model, is_training_ph=self.is_training_ph, compute_grad=False)

        Agent = {'scripted': Scripted, 'sac': SAC}[self.FLAGS['agent']]
        self.agent = Agent(sas_vals=self.sarsd_vals, sas_phs=self.sarsd_phs, embed_phs=self.embed_phs, action_dist=self.action_dist, obs_vectorizer=self.obs_vectorizer, FLAGS=FLAGS, dyn=self.dyn if self.FLAGS['share_dyn'] else None, value_dyn=self.value_dyn if self.FLAGS['aac'] else None, is_training_ph=self.is_training_ph)

        if self.FLAGS['agent'] == 'sac':
            self.scripted_agent = Scripted(sas_vals=self.sarsd_vals, sas_phs=self.sarsd_phs, embed_phs=self.embed_phs, action_dist=self.action_dist, obs_vectorizer=self.obs_vectorizer, FLAGS=FLAGS, dyn=self.dyn if self.FLAGS['share_dyn'] else None)

        self.eval_vals = {}
        if self.FLAGS['grad_summaries']:
            self.eval_vals['summary'] = tf.summary.merge(self.dyn.grad_summaries)
        else:
            self.eval_vals['summary'] = self.dyn.eval_vals.pop('summary')
            #self.eval_vals['summary'] = tf.no_op()
        self.eval_vals.update(self.dyn.eval_vals)
        self.eval_vals.update(subdict(self.dyn.step_vals, ['phi_g', 'phi_s']))
        if self.FLAGS['aac']:
            self.eval_vals.update(prefix_vals('value', self.value_dyn.eval_vals))
            self.eval_vals.update(prefix_vals('value', subdict(self.value_dyn.step_vals, ['phi_g', 'phi_s'])))
        self.eval_vals.update(subdict(self.agent.main_vals, ['v', 'q1', 'q2']))

        self.mean_rms = RunningMeanVar()
        self.sess = get_session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10, keep_checkpoint_every_n_hours=0.5)
        self.train_writer = tf.summary.FileWriter(os.path.join(self.FLAGS['log_path'], 'tb'), self.sess.graph)

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

        # SUMMARY STUFF
        self.pred_plot_ph = tf.placeholder(tf.uint8)
        self.plot_summ = tf.summary.image('mdn_contour', self.pred_plot_ph, max_outputs=3)
        self.value_plot_summ = tf.summary.image('value_mdn_contour', self.pred_plot_ph, max_outputs=3)

        self.gif_paths_ph = tf.placeholder(tf.string, shape=(None,), name='gif_path_ph')
        self.gif_summ = rl_util.gif_summary('rollout', self.gif_paths_ph, max_outputs=3)

        #self.sess.graph.finalize()
        # Make separate process for gif because it takes a long time
        self.gif_rollout_queue = Queue(maxsize=3)
        self.gif_path_queue = Queue(maxsize=3)
        # TODO: what is passed in a python process?
        # TODO: was setting this up
        if self.FLAGS['run_rl_optim']:
            self.gif_proc = Process(target=gif_plotter, daemon=True, args=(self.obs_vectorizer, self.action_dist, self.FLAGS, self.gif_rollout_queue, self.gif_path_queue))
            self.gif_proc.start()

        self.mean_summ = defaultdict(lambda: None)
        self.mean_summ_phs = defaultdict(lambda: None)

        numvars()

    def plot_contours(self, ev, itr):
        pre = 's_mdn_'
        data = plot('contour', {'state': ev[pre+'state']['array'][-1,...,:2], 'X':ev[pre+'X'], 'Y':ev[pre+'Y'], 'Z':ev[pre+'Z'][...,-1]}, self.FLAGS, itr=itr, return_data=True)
        plot_summary = self.sess.run(self.plot_summ, {self.pred_plot_ph: data[None]})
        self.train_writer.add_summary(plot_summary, global_step=itr)

        if self.FLAGS['aac']:
            pre = 'value_s_mdn_'
            data = plot('contour', {'state': ev[pre+'state']['array'][-1,...,:2], 'X':ev[pre+'X'], 'Y':ev[pre+'Y'], 'Z':ev[pre+'Z'][...,-1]}, self.FLAGS, itr=itr, return_data=True)
            plot_summary = self.sess.run(self.value_plot_summ, {self.pred_plot_ph: data[None]})
            self.train_writer.add_summary(plot_summary, global_step=itr)

    def evaluate(self, images, array=None):
        assert self.FLAGS['bs'] > 1 or not self.FLAGS['is_training']
        #assert not self.FLAGS['goal_conditioned']

        # TODO: make this names thing more robust by sorting
        idxs = self.obs_vectorizer.idxs()
        names = list(sorted(self.obs_vectorizer.dout_shape.keys()))

        img_beg, img_end = idxs[names.index('image')], idxs[names.index('image')+1]
        arr_beg, arr_end = idxs[names.index('goal_array')], idxs[names.index('goal_array')+1]
        sases = {'s': [], 'a': [], 's_next': []}
        for img in images:
            state = np.zeros(self.obs_vectorizer.out_shape)
            state[img_beg:img_end] = img.ravel()
            if array is not None:
                state[arr_beg:arr_end] = array.ravel()
            sases['s'].append(state)
            sases['a'].append(np.zeros(self.action_dist.out_shape))
            sases['s_next'].append(np.zeros(self.obs_vectorizer.out_shape))
        for key in sases: 
            sases[key] = np.stack(sases[key])
        sases['d'] = np.zeros(sases['s'].shape[0])
        sases['r'] = np.zeros(sases['s'].shape[0])

        ev = self.sess.run(self.eval_vals, self.dyn.off_policy_feed_dict(sases))
        fig, ax = plt.subplots()

        if 'vae' in self.FLAGS['cnn_gn']:
            img = ev['recon'][0]
            ax.imshow(img)
            #plt.axis('off')
            ax.axis('off')
            ax.set_frame_on(False)
            ax.margins(0,0)
            fig.tight_layout()
            #fig.patch.set_visible(False)
            #plt.savefig('4823test.png')
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            t, b, l, r = 95, 115, 115, 115
            #data = data[t:-b,l:-r,:]
            data = data.copy()
            t, b, l, r = 95, 115, 115, 115
            data = data.copy()
            data[:t,:] = 255
            data[-b:,:] = 255
            data[:,:l] = 255
            data[:,-r:] = 255
            data = data[t//2+20:-b//2-20,l//2+20:-r//2-20,:]
            #plt.cla(ax)
            plt.close(fig)
        else:
            pre = 's_mdn_'
            img = ev[pre+'state']['image'][-1]
            X, Y, Z = ev[pre+'X'], ev[pre+'Y'], np.rot90(ev[pre+'Z'][...,-1], k=3)
            ax.contour(X, Y, Z, levels=1)
            #plt.axis('off')
            ax.axis('off')
            ax.set_frame_on(False)
            ax.margins(0,0)
            fig.tight_layout()
            #fig.patch.set_visible(False)
            #plt.savefig('4823test.png')
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            #t, b, l, r = 10, 20, 40, 40
            #data = data[t:-b,l:-r,:]
            #plt.cla(ax)
            plt.close(fig)
        return data, ev

    def evaluate_array(self, array):
        idxs = self.obs_vectorizer.idxs()
        names = list(sorted(self.obs_vectorizer.dout_shape.keys()))
        array_beg, array_end = idxs[names.index('array')], idxs[names.index('array')+1]
        sases = {'s': [], 'a': [], 's_next': []}

        state = np.zeros(self.obs_vectorizer.out_shape)
        state[array_beg:array_end] = array.ravel()
        sases['s'].append(state)
        sases['a'].append(np.zeros(self.action_dist.out_shape))
        sases['s_next'].append(np.zeros(self.obs_vectorizer.out_shape))

        for key in sases: 
            sases[key] = np.stack(sases[key])
        sases['d'] = np.zeros(sases['s'].shape[0])
        sases['r'] = np.zeros(sases['s'].shape[0])

        ev = self.sess.run(self.eval_vals, self.dyn.off_policy_feed_dict(sases))
        fig, ax = plt.subplots()

        if 'vae' in self.FLAGS['cnn_gn']:
            img = ev['value_recon'][0]
            ax.imshow(img)
            #plt.axis('off')
            ax.axis('off')
            ax.set_frame_on(False)
            ax.margins(0,0)
            fig.tight_layout()
            #fig.patch.set_visible(False)
            #plt.savefig('4823test.png')
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            #t, b, l, r = map(int, input('tblr: ').split())
            t, b, l, r = 95, 115, 115, 115
            data = data.copy()
            data[:t,:] = 255
            data[-b:,:] = 255
            data[:,:l] = 255
            data[:,-r:] = 255
            data = data[t//2+20:-b//2-20,l//2+20:-r//2-20,:]
            #plt.cla(ax)
            plt.close(fig)
        else:
            pre = 'value_s_mdn_'
            #import ipdb; ipdb.set_trace()
            img = ev[pre+'state']['image'][-1]
            X, Y, Z = ev[pre+'X'], ev[pre+'Y'], np.rot90(ev[pre+'Z'][...,-1], k=3)
            ax.contour(X, Y, Z, levels=1)
            ax.scatter(-array[:,1], array[:,0])
            ax.axis('off')
            ax.set_frame_on(False)
            ax.margins(0,0)
            fig.tight_layout()
            #plt.savefig('test.png')
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            #t, b, l, r = map(int, input('tblr: ').split())
            #t, b, l, r = 95, 115, 115, 115
            #data = data[t:-b,l:-r,:]
            #plt.cla(ax)
            plt.close(fig)
        return data, ev

    def query_policy_array(self, current_image, goal_array):
        """Take in current image and goal image and produce action and info"""
        obs = {key: np.zeros(val, np.float32) for key, val in self.obs_vectorizer.dout_shape.items()}
        obs['image'] = current_image
        obs['goal_array'] = goal_array

        compute_vals = lambda obs: tuple([obs[key] for key in sorted(obs.keys())])
        vals = compute_vals(obs)
        model_out = self.agent.model.step([vals], None, mode='explore')
        info = {}
        return model_out['actions'][0], info

    def query_policy(self, current_image, goal_image):
        """Take in current image and goal image and produce action and info"""
        obs = {key: np.zeros(val, np.float32) for key, val in self.obs_vectorizer.dout_shape.items()}
        obs['image'] = current_image
        obs['goal_image'] = goal_image

        compute_vals = lambda obs: tuple([obs[key] for key in sorted(obs.keys())])
        vals = compute_vals(obs)
        model_out = self.agent.model.step([vals], None, mode='explore')
        vec_obs = self.obs_vectorizer.to_vecs([vals])

        if self.FLAGS['aac']:
            dyn_vals = self.value_dyn.step({'s': vec_obs, 'a': model_out['actions'], 's_next':vec_obs})
            dist = cos_dist(dyn_vals['phi_s'][0], dyn_vals['phi_g'][0])
            info = {'dist': dist}
        else:
            info = {}

        return model_out['actions'][0], info
