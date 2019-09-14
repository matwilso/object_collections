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
from anyrl.envs import batched_gym_env
from anyrl.models import MLP

from baselines import logger
from gym.spaces import Tuple

import object_collections.rl.util as rl_util
from object_collections.envs import TABLES, make_fn
from object_collections.sl.viz import image_tile_summary, plot
from object_collections.rl.agents import Scripted, SAC
from object_collections.rl.encoders import DYN, VAE
from object_collections.rl.mpi_util import RunningMeanVar
from object_collections.tf_util import get_session

from .data import rollout_to_pickle, rollout_to_tf_record
from . import rollers

def prefix_vals(name, vals): return {name.lower() + '_' + key: vals[key] for key in vals} # add a prefix based on the name (agent or encoder)
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

# TODO: fix stuff to not be called dyn in code that this calls

class Trainer:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.is_training_ph = tf.placeholder(tf.bool, [])

        with rl_util.Timer('building_envs', self.FLAGS):
            fns = [make_fn(i, FLAGS) for i in range(FLAGS['num_envs'])]
            if self.FLAGS['num_envs'] == 1:
                self.env = fns[0]()
            else:
                self.env = batched_gym_env(fns)

        self.action_dist = rl_util.convert_to_dict_dist(self.env.action_space.spaces)
        self.obs_vectorizer = rl_util.convert_to_dict_dist(self.env.observation_space.spaces)

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


        # ALGO SETUP
        Encoder = VAE if 'vae' in self.FLAGS['cnn_gn'] else DYN
        EName = 'VAE' if 'vae' in self.FLAGS['cnn_gn'] else 'DYN'

        if self.FLAGS['goal_dyn'] != '':
            self.goal_model = DYN(self.sarsd_vals, self.sarsd_phs, self.action_dist, self.obs_vectorizer, FLAGS, conv='cnn' in self.FLAGS['goal_dyn'], name='GoalDYN', compute_grad=False).model
        else:
            self.goal_model = None

        if self.FLAGS['aac']:
            self.value_encoder = Encoder(self.sarsd_vals, self.sarsd_phs, self.action_dist, self.obs_vectorizer, FLAGS, conv=False, name='Value'+EName)
            if self.FLAGS['value_goal']:
                self.goal_model = self.value_encoder.model
        else:
            self.value_encoder = None

        self.encoder = Encoder(self.sarsd_vals, self.sarsd_phs, self.action_dist, self.obs_vectorizer, FLAGS, conv='cnn' in self.FLAGS['cnn_gn'], goal_model=self.goal_model, is_training_ph=self.is_training_ph)


        Agent = {'scripted': Scripted, 'sac': SAC}[self.FLAGS['agent']]
        self.agent = Agent(sas_vals=self.sarsd_vals, sas_phs=self.sarsd_phs, embed_phs=self.embed_phs, action_dist=self.action_dist, obs_vectorizer=self.obs_vectorizer, FLAGS=FLAGS, dyn=self.encoder if self.FLAGS['share_dyn'] else None, value_dyn=self.value_encoder if self.FLAGS['aac'] else None, is_training_ph=self.is_training_ph)

        if self.FLAGS['agent'] == 'sac':
            self.scripted_agent = Scripted(sas_vals=self.sarsd_vals, sas_phs=self.sarsd_phs, embed_phs=self.embed_phs, action_dist=self.action_dist, obs_vectorizer=self.obs_vectorizer, FLAGS=FLAGS, dyn=self.encoder if self.FLAGS['share_dyn'] else None)

        self.eval_vals = {}
        if self.FLAGS['grad_summaries']:
            self.eval_vals['summary'] = tf.summary.merge(self.encoder.grad_summaries)
        else:
            self.eval_vals['summary'] = self.encoder.eval_vals.pop('summary')
            #self.eval_vals['summary'] = tf.no_op()
        self.eval_vals.update(self.encoder.eval_vals)
        if self.FLAGS['aac']:
            self.eval_vals.update(prefix_vals('value', self.value_encoder.eval_vals))

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
            #import ipdb; ipdb.set_trace()
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
        if self.FLAGS['run_rl_optim']:
            self.gif_proc = Process(target=gif_plotter, daemon=True, args=(self.obs_vectorizer, self.action_dist, self.FLAGS, self.gif_rollout_queue, self.gif_path_queue))
            self.gif_proc.start()

        self.mean_summ = defaultdict(lambda: None)
        self.mean_summ_phs = defaultdict(lambda: None)


        numvars()

        if self.FLAGS['threading']:
            # multi-thread for a moderate speed-up
            self.rollout_queue = queue.Queue(maxsize=3)
            self.rollout_thread = Thread(target=self.rollout_maker, daemon=True)

    def rollout_maker(self):
        raise NotImplementedError()

    def log(self, means, itr):
        for key in means:
            if self.mean_summ_phs[key] is None:
                self.mean_summ_phs[key] = tf.placeholder(np.array(means[key]).dtype, name=key+'_ph')
            if self.mean_summ[key] is None:
                self.mean_summ[key] = self.mean_summ[key] or tf.summary.scalar(key, self.mean_summ_phs[key])

        feed_dict = {self.mean_summ_phs[key]: np.mean(means[key]) for key in means}
        summ = self.sess.run(subdict(self.mean_summ, means.keys()), feed_dict)
        for key in means.keys(): self.train_writer.add_summary(summ[key], global_step=itr)
        self.means = defaultdict(lambda: [])

        # Grab any new gif paths floating around and plot them
        while not self.gif_path_queue.empty():
            paths = self.gif_path_queue.get()
            if self.FLAGS['debug']:
                print('GOT PATHS', paths)
            gif_summ = self.sess.run(self.gif_summ, {self.gif_paths_ph: paths})
            self.train_writer.add_summary(gif_summ, global_step=itr)


    def plot_contours(self, ev, itr):
        if self.FLAGS['aac']:
            pre = 'value_s_mdn_'
            data = plot('contour', {'state': ev[pre+'state']['array'][-1,...,:2], 'X':ev[pre+'X'], 'Y':ev[pre+'Y'], 'Z':ev[pre+'Z'][...,-1]}, self.FLAGS, itr=itr, return_data=True)
            plot_summary = self.sess.run(self.value_plot_summ, {self.pred_plot_ph: data[None]})
            self.train_writer.add_summary(plot_summary, global_step=itr)

        pre = 's_mdn_'
        data = plot('contour', {'state': ev[pre+'state']['array'][-1,...,:2], 'X':ev[pre+'X'], 'Y':ev[pre+'Y'], 'Z':ev[pre+'Z'][...,-1]}, self.FLAGS, itr=itr, return_data=True)
        plot_summary = self.sess.run(self.plot_summ, {self.pred_plot_ph: data[None]})
        self.train_writer.add_summary(plot_summary, global_step=itr)



class OffPolicyTrainer(Trainer):
    def __init__(self, FLAGS):
        super().__init__(FLAGS)
        self.sess.run(self.agent.init_op)

        if self.FLAGS['num_envs'] == 1:
            self.roller = rollers.BasicRoller(self.env, self.agent.model, min_episodes=2, min_steps=10*self.FLAGS['horizon'])
        else:
            self.roller = rollers.EpisodeRoller(self.env, self.agent.model)
        
        assert self.FLAGS['max_episode_steps'] >= 1

    def eval_policy(self, itr, myag=True):
        env_stats = {}

        if itr < self.FLAGS['agent_burn_in']:
            rollouts = self.eval_roller.rollouts(model=self.scripted_agent.model)
        else:
            rollouts = self.eval_roller.rollouts(mode='explore')

        # OG rollouts
        rollouts, sarsd, dyn_vals, extra_info = rl_util.dyn_post_proc(rollouts, self.action_dist, self.obs_vectorizer, self.encoder, self.FLAGS, value_dyn=self.value_encoder)
        dgoals = sarsd['dgoal']
        sarsd = flatten_list_dict(sarsd)

        env_stats['rews'] = np.mean(sarsd['r'])
        env_stats['abs_rews'] = np.mean(np.abs(sarsd['r']))
        env_stats.update(extra_info)

        eval_vals = self.sess.run(self.eval_vals, {**self.encoder.off_policy_feed_dict(sarsd), self.is_training_ph: self.FLAGS['is_training']})
        if 'vae' not in self.FLAGS['cnn_gn']:
            self.plot_contours(eval_vals, itr)
        else:
            self.train_writer.add_summary(eval_vals['summary'], global_step=itr)

        rlen = len(rollouts)

        if myag and self.FLAGS['goal_conditioned'] and self.FLAGS['agent'] != 'scripted' :
            pre = 'value_phi_' if self.FLAGS['aac'] else 'phi_'
            #dgoals = [np.linalg.norm(dyn_vals[pre+'s'][i] - dyn_vals[pre+'g'][i], axis=-1) for i in range(rlen)]
            simis = []
            for i in range(rlen):
                simi = [cos_dist(dyn_vals[pre+'s'][i][j], dyn_vals[pre+'g'][i][j]) for j in range(len(dyn_vals[pre+'s'][i]))]
                simis.append(simi)

            if not self.gif_rollout_queue.full():
                self.gif_rollout_queue.put((itr, rollouts, dgoals, simis))
        means = prefix_vals('eval', {key: np.mean(env_stats[key]) for key in env_stats})
        return means

    def rollout_maker(self):
        while True:
            if self.rollout_kill:
                return
            start = time.time()
            rollouts = self.roller.rollouts(**self.explore_args(self.itr))
            dt = time.time() - start
            self.rollout_queue.put((dt, rollouts))

    def explore_args(self, itr):
        if self.FLAGS['explore_anneal'] > 0:
            slope = (self.FLAGS['explore_frac'] - 1.0) / self.FLAGS['explore_anneal']
            anneal = 1.0 + slope * itr
            frac = max(self.FLAGS['explore_frac'], anneal)
        else:
            frac = self.FLAGS['explore_frac']
        p = frac
        args = {'model': self.agent.model, 'p': p, 'scripted': self.scripted_agent.model}
        return args

    def run(self):
        print('running algorithm')
        self.eval_env = make_fn(420, self.FLAGS)()
        self.eval_roller = rollers.BasicRoller(self.eval_env, self.agent.model, min_episodes=3, min_steps=10*self.FLAGS['horizon'])
        self.itr = self.sess.run(self.global_itr)
        last_iter = time.time()

        means = defaultdict(lambda: [])
        out = {}
        replay_buff = rollers.ReplayBuffer(self.obs_vectorizer.out_shape[0], self.action_dist.out_shape[0], self.FLAGS)

        if self.FLAGS['threading']:
            self.rollout_kill = False
            self.rollout_thread.start()

        while True:
            with rl_util.Timer('rollout_time', {'debug': 0}) as rollout_timer:
                if self.FLAGS['threading']:
                    if self.rollout_queue.empty():
                        rollouts = None
                    else:
                        dt_rollout, rollouts = self.rollout_queue.get()
                        if self.FLAGS['debug']: print('dt_rollout (in other thread)', dt_rollout)

                    if self.FLAGS['debug'] and self.rollout_queue.qsize() != 0: print('qsize', self.rollout_queue.qsize())
                else:
                        rollouts = self.roller.rollouts(**self.explore_args(self.itr))
            if rollouts is not None:
                means['dt_waiting'] += [rollout_timer.interval]
                if self.FLAGS['debug']: rollout_timer.display()

            if rollouts is not None:
                rollouts, sarsd, dyn_vals, extra_info = rl_util.dyn_post_proc(rollouts, self.action_dist, self.obs_vectorizer, self.encoder, self.FLAGS, value_dyn=self.value_encoder)
                for key in extra_info: means[key] += [extra_info[key]]
                if self.FLAGS['use_her']:
                    # HER rollouts
                    her_sarsd, her_extra_info = rl_util.herify(sarsd, dyn_vals, self.obs_vectorizer, FLAGS=self.FLAGS)
                    sarsd = flatten_list_dict(sarsd)
                    sarsd = {key: np.concatenate([sarsd[key], her_sarsd[key]]) for key in her_sarsd}
                    means['herr'] += [np.mean(her_sarsd['r'])]
                    for key in her_extra_info: means[key] += [her_extra_info[key]]
                else:
                    sarsd = flatten_list_dict(sarsd)

                replay_buff.store_batch(sarsd)

            # TODO: move this training logic into the algorithm stuff.
            # the only thing that should be in this file is like tensorboard logging and stuff

            if replay_buff.size >= self.FLAGS['min_replay_size']:
                with rl_util.Timer('optim_step', self.FLAGS) as optim_timer:
                    for train_idx in range(self.FLAGS['optim_steps']):
                        batch = replay_buff.sample_batch(self.FLAGS['bs'], self.FLAGS['phi_noise'])
                        main_out = self.sess.run(self.agent.main_vals, {**self.agent.feed_dict(batch), self.is_training_ph: self.FLAGS['is_training']})
                        for key in main_out:
                            if main_out[key] is not None:
                                means[key].append(np.mean(main_out[key]))
                means['dt_optim'] += [optim_timer.interval]

                if rollouts is not None:
                    means['rews'] = mean_reward(rollouts)
                    means['abs_rews'] = abs_mean_reward(rollouts)
                    means['rets'] = mean_return(rollouts)
                    means['abs_rets'] = abs_mean_return(rollouts)

                #if self.itr % self.FLAGS['eval_n'] == 0:
                if self.itr % self.FLAGS['eval_n'] == 0:
                    for key in means: means[key] = np.mean(means[key])
                    with rl_util.Timer('eval_policy', self.FLAGS) as dt_eval:
                        means.update(self.eval_policy(itr=self.itr))
                    means['dt_eval'] += [dt_eval.interval]
                    if self.FLAGS['use_her']:
                        print('itr {0:.4f}: abs_rew_mean {1:.4f} eps_rew_mean {2:.4f} her {3:.4f} frac {4:.4f}'.format(self.itr, means['eval_abs_rews'], means['eval_rews'], means['herr'], means['eval_reached_frac']))
                    else:
                        print('itr {0:.4f}: abs_rew_mean {1:.4f} eps_rew_mean {2:.4f} frac {3:.4f}'.format(self.itr, means['eval_abs_rews'], means['eval_rews'], means['eval_reached_frac']))

                    self.log(means, self.itr)
                    means = defaultdict(lambda: [])

                if self.itr % self.FLAGS['save_n'] == 0:
                    self.saver.save(self.sess, self.FLAGS['save_path'], global_step=self.itr)
                    #save_vars(self.sess, self.FLAGS['save_path'])
                    print('saved vars', self.FLAGS['save_path'])
                    #with open('replay_buff{}.pkl'.format(self.FLAGS['suffix']), 'wb') as f:
                    #    cloudpickle.dump(replay_buff, f)

                self.itr = self.sess.run(self.inc_global)
                now = time.time()
                means['dt_total'] += [now - last_iter]
                if self.FLAGS['debug']:
                    print('dt', now-last_iter)
                last_iter = now

                if self.itr >= self.FLAGS['total_n']:
                    p1 = self.eval_policy(itr=self.itr, myag=False)
                    p2 = self.eval_policy(itr=self.itr, myag=False)
                    p3 = self.eval_policy(itr=self.itr, myag=False)
                    for key in p1: out[key] = np.mean([p1[key], p2[key], p3[key]])
                    break

        if self.FLAGS['threading']:
            self.rollout_kill = True
            if not self.rollout_queue.empty():
                self.rollout_queue.get()
            self.rollout_thread.join()
        return out

class OnPolicyTrainer(Trainer):
    """
    Class from when we were trying on-policy methods.
    Stays in use for scripted policy data collection.
    """

    def __init__(self, FLAGS):
        super().__init__(FLAGS)
        # vals are things that rl algos report, like loss and feature_var and stuff
        self.vals = {**prefix_vals(self.agent.name, self.agent.vals), **prefix_vals(self.encoder.name, self.encoder.vals)}

        if self.FLAGS['num_envs'] == 1:
            self.roller = rollers.BasicRoller(self.env, self.agent.model, min_episodes=2, min_steps=self.FLAGS['horizon'])
        else:
            self.roller = rollers.TruncatedRoller(self.env, self.agent.model, num_timesteps=self.FLAGS['horizon'])

        self.num_iter = (self.FLAGS['num_epochs'] * (self.FLAGS['num_envs'] * self.FLAGS['horizon'])) / self.FLAGS['bs']  # = (K*NT) / M

    def inner_loop(self, itr, rollouts):
        with rl_util.Timer('dyn_post_proc', self.FLAGS):
            rollouts, sarsd, dyn_vals, extra_info = rl_util.dyn_post_proc(rollouts, self.action_dist, self.obs_vectorizer, self.encoder, self.FLAGS, value_dyn=self.value_encoder)

        # Plot gifs of rollouts
        if itr % self.FLAGS['eval_n'] == 0:
            idxs = np.random.choice(range(len(rollouts)), size=min(3, self.FLAGS['num_envs']), replace=False) 

            gif_rollouts = [rollouts[i] for i in idxs]
            if self.FLAGS['goal_conditioned'] and self.FLAGS['agent'] != 'scripted':
                dgoals = [np.linalg.norm(dyn_vals['phi_s'][i] - dyn_vals['phi_g'][i], axis=-1) for i in idxs]
                simis = []
                for idx in idxs:
                    simis.append([cos_dist(dyn_vals['phi_s'][idx][j], dyn_vals['phi_g'][idx][j]) for j in range(len(dyn_vals['phi_s'][idx]))])
                if not self.gif_rollout_queue.full():
                    self.gif_rollout_queue.put((itr, gif_rollouts, dgoals, simis))
            else:
                if not self.gif_rollout_queue.full():
                    self.gif_rollout_queue.put((itr, gif_rollouts, None, None))

        with rl_util.Timer('training', self.FLAGS):
            advantages = self.agent.adv_est.advantages(rollouts)
            targets = self.agent.adv_est.targets(rollouts)

            # NOTE: this must be called after the post-proc on the rollouts
            batches = rl_util.batches(rollouts, self.obs_vectorizer, batch_size=self.FLAGS['bs'])

            # TODO: is it a problem that sarsd now has a different shape because we are shortening the rollouts
            # can we leave them the same length or do we need to fix them?
            # Seems like we can leave them since we are indexing later by timestep idx (in batch)
            assert len(advantages[0]) == len(targets[0]) and len(advantages[0]) == len(rollouts[0].rewards)

            agent_key = 'post_burn' if itr >= self.FLAGS['agent_burn_in'] or self.FLAGS['share_dyn'] else 'burn_in'

            # Determine what to train
            train_ops = {}
            if self.FLAGS['agent'] == 'scripted':
                pass
            else:
                train_ops['encoder'] = self.encoder.train_op
                train_ops['encoder'] = self.encoder.global_step
                train_ops['agent'] = self.agent.train_op[agent_key]
                train_ops['agent_step'] = self.agent.global_step

                if self.FLAGS['goal_conditioned']:
                    # don't update the features in this case. keep them frozen 
                    train_ops.pop('encoder')
                    train_ops.pop('encoder_step')
                elif itr < self.FLAGS['dyn_warm_start']:
                    # only run the curiosity training because it usually starts extremely bad
                    train_ops.pop('agent')
                    train_ops.pop('agent_step')

            batch_idx = 0
            for batch in batches:
                feed_dict = {**self.agent.feed_dict(rollouts, batch, advantages, targets), **self.encoder.feed_dict(rollouts, batch, sarsd, dyn_vals)}

                if batch_idx < self.num_iter:
                    self.sess.run(train_ops, feed_dict=feed_dict)
                    batch_idx += 1
                else:
                    # grab values for last step
                    vals, tops = self.sess.run([self.vals, train_ops], feed_dict=feed_dict)
                    vals.update({key: tops[key] for key in ['encoder_step', 'agent_step'] if key in tops})

                    # and sometimes grab summaries
                    if itr % self.FLAGS['eval_n'] == 0 and self.FLAGS['mdn_aux']:
                        eval_vals = self.sess.run(self.eval_vals, feed_dict={**feed_dict, self.is_training_ph: self.FLAGS['is_training']})
                        self.train_writer.add_summary(eval_vals['summary'], global_step=itr)
                        with rl_util.Timer('plot_contours', self.FLAGS):
                            try:
                                if 'vae' not in self.FLAGS['cnn_gn']:
                                    self.plot_contours(eval_vals, itr)
                            except:
                                print('plot_countours crash')
                        self.train_writer.flush()

                    break
        return vals, rollouts

    def run(self):
        if self.FLAGS['threading']:
            self.rollout_kill = False
            self.rollout_thread.start()

        print('running algorithm')
        self.means = defaultdict(lambda: [])

        itr = self.sess.run(self.global_itr)
        while True:
            dt_total = rl_util.Timer('total_time', self.FLAGS)
            dt_total.__enter__()  # have to do hacky enter and exit since the indent levels don't line up
            with rl_util.Timer('rollout_time', self.FLAGS) as dt_waiting:
                if self.FLAGS['threading']:
                    dt_rollout, rollouts = self.rollout_queue.get()
                    self.means['dt_rollout'] += [dt_rollout]
                else:
                    rollouts = self.roller.rollouts()

            self.means['dt_waiting'] += [dt_waiting.interval]
            if self.FLAGS['run_rl_optim']:
                with rl_util.Timer('inner_loop', self.FLAGS) as dt_inner:
                    result, rollouts = self.inner_loop(itr, rollouts)
                dt_total.__exit__()

                self.means['dt_inner'] += [dt_inner.interval]
                self.means['dt_total'] += [dt_total.interval]

                if self.FLAGS['goal_conditioned'] or itr > self.FLAGS['encoder_warm_start']:
                    self.means['ep_rew_mean'] += [mean_return(rollouts)]
                    self.means['abs_rew_mean'] += [abs_mean_return(rollouts)]
                    for key in result.keys():
                        self.means[key] += [np.mean(result[key])]

                    if itr % self.FLAGS['eval_n'] == 0:
                        print('itr {0:.4f}: abs_rew_mean {1:.4f} eps_rew_mean {2:.4f} dt {3:.4f}'.format(itr, np.mean(self.means['abs_rew_mean']), np.mean(self.means['ep_rew_mean']), np.mean(self.means['dt_total'])))
                        self.log(self.means, itr=itr)


                if itr % self.FLAGS['save_n'] == 0:
                    self.saver.save(self.sess, self.FLAGS['save_path'], global_step=itr)
                    #save_vars(self.sess, self.FLAGS['save_path'])
                    print('saved vars', self.FLAGS['save_path'])
            else:
                dt_total.__exit__()
                print('itr {}, dt {}'.format(itr, np.mean(self.means['dt_waiting'])))
                self.log(self.means, itr=itr)
                
            if self.FLAGS['dump_rollouts']:
                self.dump_rollouts(rollouts)
            itr = self.sess.run(self.inc_global)
            if itr >= self.FLAGS['total_n']:
                return
                
        if self.FLAGS['threading']:
            self.rollout_kill = True
            if not self.rollout_queue.empty():
                self.rollout_queue.get()
            self.rollout_thread.join()

    def rollout_maker(self):
        while True:
            if self.rollout_kill:
                return
            start = time.time()
            rollouts = self.roller.rollouts()
            dt = time.time() - start
            self.rollout_queue.put((dt, rollouts))

    def dump_rollouts(self, rollouts):
        if self.FLAGS['goal_conditioned']:
            rollouts, sarsd, dyn_vals, extra_info = rl_util.dyn_post_proc(rollouts, self.action_dist, self.obs_vectorizer, self.encoder, self.FLAGS, value_dyn=self.value_encoder)
        else:
            rollouts, sarsd, dyn_vals, extra_info = rl_util.dyn_post_proc(rollouts, self.action_dist, self.obs_vectorizer, None, self.FLAGS, value_dyn=self.value_encoder)

        rollout_to_tf_record(sarsd, self.obs_vectorizer, self.FLAGS)
        #rollout_to_pickle(rollouts, sarsd, self.FLAGS)


def gif_plotter(obs_vectorizer, action_dist, FLAGS, gif_rollout_queue, gif_path_queue):
    while True:
        if not gif_rollout_queue.empty():
            while gif_rollout_queue.qsize() > 1:
                gif_rollout_queue.get()  # Throw extra away if we can't keep up

            gif_vals = gif_rollout_queue.get()
            paths = rl_util.plot_rollout(obs_vectorizer, action_dist, FLAGS, *gif_vals)
            gif_path_queue.put(paths)
        time.sleep(0.1)
