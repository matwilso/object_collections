import copy
import os
import itertools
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.spatial.distance import cosine as cos_dist
import PIL
import tensorflow as tf
from anyrl.envs.wrappers import BatchedWrapper
from anyrl.spaces import gym_space_distribution, gym_space_vectorizer
from anyrl.spaces.aggregate import TupleDistribution
from matplotlib import animation
from tensorflow.core.framework import summary_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io.tf_record import (TFRecordCompressionType,
                                                TFRecordOptions,
                                                TFRecordWriter)
from tensorflow.python.ops import summary_op_util, summary_ops_v2
from tensorflow_probability import distributions as tfd

from graph_nets.graphs import GraphsTuple
from object_collections.envs import TABLES, unmap_continuous
def subdict(dict, keys): return {key: dict[key] for key in keys}
def prefix_vals(name, vals): return {name.lower() + '_' + key: vals[key] for key in vals} # add a prefix based on the name (agent or dyn)
def suffix_vals(name, vals): return {key + '_' + name.lower(): vals[key] for key in vals}
from object_collections.sl.util import convert_nodes_to_graph_tuple
from anyrl.rollouts import empty_rollout
from object_collections.tf_util import get_session
from object_collections.sl.aux_losses import mask_state

def flatten_mdn(logits, locs, scales, FLAGS):
    """
    logits = (BS x MDN_K)
    locs = (BS x MDN_K x DIM)
    scales = (BS x MDN_K x DIM)
    """
    locs = tf.reshape(locs, [-1, FLAGS['mdn_k']*FLAGS['DIM']])
    scales = tf.reshape(scales, [-1, FLAGS['mdn_k']*FLAGS['DIM']])
    return tf.concat([logits, locs, scales], axis=1)

def np_unflatten_mdn(combined, FLAGS):
    return _unflatten_mdn(np, combined, FLAGS)

def tf_unflatten_mdn(combined, FLAGS):
    return _unflatten_mdn(tf, combined, FLAGS)

def _unflatten_mdn(rs, combined, FLAGS):
    """
    combined = (BS X [mdn_k + mdn_k*dim*2])
    """
    # TODO: write test to make sure same in = same out for these 2 funcs
    logits = combined[:,:FLAGS['mdn_k']]
    locs = combined[:,FLAGS['mdn_k']:FLAGS['mdn_k']+FLAGS['mdn_k']*FLAGS['DIM']]
    scales = combined[:,FLAGS['mdn_k']+FLAGS['mdn_k']*FLAGS['DIM']:]
    locs = rs.reshape(locs, [-1, FLAGS['mdn_k'], FLAGS['DIM']])
    scales = rs.reshape(scales, [-1, FLAGS['mdn_k'], FLAGS['DIM']])
    return logits, locs, scales

CACHE = {'MDN': None}
def compute_mdn_probs(logits, locs, scales, array, FLAGS):
    """compute probability of all blocks being in correct spots"""
    if CACHE['MDN'] is None:
        CACHE['logits_ph'] = tf.placeholder(tf.float32, shape=[None, *logits.shape[1:]], name='logits_ph')
        CACHE['locs_ph'] = tf.placeholder(tf.float32, shape=[None, *locs.shape[1:]], name='locs_ph')
        CACHE['scales_ph'] = tf.placeholder(tf.float32, shape=[None, *scales.shape[1:]], name='scales_ph')
        CACHE['array_ph'] = tf.placeholder(tf.float32, shape=[None, *array.shape[1:]], name='array_ph')
        cat = tfd.Categorical(logits=logits)
        comp = tfd.MultivariateNormalDiag(loc=locs, scale_diag=scales)
        mixture = tfd.MixtureSameFamily(cat, comp)
        CACHE['MDN'] = mixture
        mask0, mask0_count, tstate = mask_state({'array':CACHE['array_ph']})
        # TODO: add mask state logic here if needed
        # TODO: probably want to get rid of masked ones and scale by number of objects to make it more balanced for more vs. less objects
        #CACHE['prob_op'] = mixture.prob(tstate)
        CACHE['log_prob_op'] = mixture.log_prob(tstate)

    sess = get_session()
    feed_dict = {CACHE['logits_ph']: logits, CACHE['locs_ph']: locs, CACHE['scales_ph']: scales, CACHE['array_ph']: array}
    log_prob = sess.run(CACHE['log_prob_op'], feed_dict)
    combined_log_prob = np.sum(log_prob, axis=0) # combine along number of shapes
    return -combined_log_prob

#def np_mdn(logits, locs, scales, FLAGS):
#    from sklearn.mixture import GaussianMixture
#    gm = GaussianMixture(n_components=FLAGS['mdn_k'], covariance_type='diag', weights_init=logits, means_init=locs, precisions_init=1.0/(scales + 1e-9)) 
#    return gm

def sarsd_to_vals(sarsd, obs_vectorizer, FLAGS):
    """Take just current state as state"""
    sarsd_vals = {}
    s = obs_vectorizer.to_tf_dict(sarsd['s'])
    s_next = obs_vectorizer.to_tf_dict(sarsd['s_next'])

    sarsd_vals['s'] = {'graph': convert_nodes_to_graph_tuple(s['array'], FLAGS), 'array': s['array']}
    if FLAGS['use_image']: 
        sarsd_vals['s']['image'] = s['image']
        if FLAGS['use_canonical']:
            sarsd_vals['s']['canonical'] = s['canonical']

    sarsd_vals['s_next'] = {'graph': convert_nodes_to_graph_tuple(s_next['array'], FLAGS), 'array': s_next['array']}
    if FLAGS['use_image']: 
        sarsd_vals['s_next']['image'] = s_next['image']
        if FLAGS['use_canonical']:
            sarsd_vals['s_next']['canonical'] = s_next['canonical']

    sarsd_vals['a'] = sarsd['a']
    if 'r' in sarsd:
        sarsd_vals['r'] = sarsd['r']
        sarsd_vals['d'] = sarsd['d']
    else:
        sarsd_vals['r'] = tf.zeros_like(sarsd['s'][:,0])
        sarsd_vals['d'] = tf.zeros_like(sarsd['s'][:,0])

    if FLAGS['goal_conditioned']:
        g_array = s['goal_array']
        g_state = {'graph': convert_nodes_to_graph_tuple(g_array, FLAGS), 'array': g_array}
        if FLAGS['use_image']: g_state['image'] = s['goal_image']
        sarsd_vals['g'] = g_state

    return sarsd_vals

def transitions_to_rollout(transitions):
    rollout = empty_rollout(None)
    for t in transitions:
        rollout.observations.append(t['obs'])
        rollout.rewards.append(t['rewards'][0])
        rollout.model_outs.append(t['model_outs'])
        rollout.infos.append(t['info'])

        if t['is_last']:
            break
    return rollout

# TODO: get rid of support for dense reward to make this simpler
def swap_rewards(rollouts, sarsd, dyn_vals, obs_vectorizer, FLAGS):
    """
    Swap rewards in rollouts for rewards in new_rewards and replace information to make things consistent.
    NOTE: this reduces the lengths of every rollout by 1

    Also does some other reward post-processing

    Args:
        add_old (bool): True to combine the two rewards into a single signal
    """
    new_rewards = [np.zeros(rs.shape[0], np.float32) for rs in dyn_vals['phi_s']]

    assert len(sarsd['s']) == len(new_rewards)

    new_sarsd = {key: [] for key in list(sarsd.keys()) + ['og_reward', 'dgoal']}
    
    new_dyn_vals = {key: [] for key in dyn_vals.keys()}
    new_rollouts = []
    reached_goals = []

    for ridx in range(len(sarsd['s'])):
        og_rewards = np.array(sarsd['r'][ridx][:len(new_rewards[ridx])])
        new_rewards[ridx] += FLAGS['old_weight'] * og_rewards

        s_array, s_next_array, goal_array = obs_vectorizer.to_np_dict(sarsd['s'][ridx])['array'], obs_vectorizer.to_np_dict(sarsd['s_next'][ridx])['array'], obs_vectorizer.to_np_dict(sarsd['s'][ridx])['goal_array']
        if FLAGS['penalize_stasis']:
            # Penalize the agent if it does not move
            deltas = (s_next_array - s_array)
            sum_deltas = np.sum(np.abs(deltas), axis=(-1,-2))
            assert sum_deltas.shape == new_rewards[ridx].shape
            stasis_mask = (sum_deltas < FLAGS['stasis_threshold'])
            move_mask = np.logical_not(stasis_mask)
            new_rewards[ridx][move_mask] += FLAGS['stasis_rew']
            new_rewards[ridx][stasis_mask] += -FLAGS['stasis_rew']

        if FLAGS['goal_conditioned'] and not FLAGS['only_stasis']:
            if FLAGS['aac']:
                # compute rewards using value (GN) based embedding.
                # i think this would help the reward signal be a bit smoother than image
                phi_s = dyn_vals['value_reward_phi_s'][ridx] 
                phi_s_next = dyn_vals['value_reward_phi_s_next'][ridx] 
                phi_g = dyn_vals['value_reward_phi_g'][ridx]
            else:
                phi_s = dyn_vals['reward_phi_s'][ridx] 
                phi_s_next = dyn_vals['reward_phi_s_next'][ridx] 
                phi_g = dyn_vals['reward_phi_g'][ridx]

            dgoal = np.array([cos_dist(phi_g[i], phi_s_next[i]) for i in range(len(phi_g))])

            new_sarsd['dgoal'].append(dgoal)
            achieved_goal = dgoal < FLAGS['goal_threshold']
            done_idx = np.argmax(achieved_goal)

            # TODO: really fix this so that we have a consistent reward function.  It should 
            # just be a function that takes in certain things so that we can reuse it.

            # Everything before goal is -1 (including if we don't reach goal)
            # at goal, reward = 0.0
            if done_idx == 0:
                if achieved_goal[0] > 0.0:
                    # This means get rid of rollouts if we are already at the goal
                    continue 
                else:
                    # This means we never reached it, so all steps get -1
                    new_rewards[ridx][:] += -1.0
                reached_goals.append(0)
            else:
                # This means we reached goal at some point
                # Shorten rollout to end at goal and give rewards out
                new_rewards[ridx] = new_rewards[ridx][:done_idx+1]
                new_rewards[ridx][:done_idx] += -1.0
                # Test here to only give good reward if they actually moved
                # NOTE: this probably is not required since I think it is impossible to get here without HER, but at least this makes things consisten
                if new_rewards[ridx][done_idx] > 0.0:
                    new_rewards[ridx][done_idx] += 1.0
                reached_goals.append(1)


            sarsd['r'][ridx] = new_rewards[ridx]

        new_len = len(new_rewards[ridx])

        # store new rewards in sarsd
        sarsd['r'][ridx] = new_rewards[ridx]

        # Adapt length of sarsd and dyn vals so they will match up downstream
        for key in sarsd: 
            new_sarsd[key].append(sarsd[key][ridx][:new_len])
        new_sarsd['og_reward'].append(og_rewards)
        for key in new_dyn_vals: 
            new_dyn_vals[key].append(dyn_vals[key][ridx][:new_len])

        # Fix the shape of rollouts
        rollouts[ridx].rewards = new_rewards[ridx]
        # correct the length of the other settings so anyrl.rollouts doesn't bark at us
        rollouts[ridx].infos = rollouts[ridx].infos[:new_len]
        rollouts[ridx].model_outs = rollouts[ridx].model_outs[:new_len]
        rollouts[ridx].observations = rollouts[ridx].observations[:new_len]
        # reset this because it does not make sense for endless envs
        rollouts[ridx].prev_reward = 0.0
        new_rollouts.append(rollouts[ridx])

    return new_rollouts, new_sarsd, new_dyn_vals, {'reached_frac': np.mean(reached_goals)}

def dyn_post_proc(rollouts, action_dist, obs_vectorizer, dyn, FLAGS, value_dyn=None):
    """"""
    actvec = lambda x: action_dist.to_vecs(x)
    append = lambda alls, currs: [alls[key].append(currs[key]) for key in currs]

    keys = ['s', 'a', 's_next', 'r', 'd']
    if FLAGS['use_embed']: 
        keys += FLAGS['embeds']
        keys += ['reward_' + key for key in FLAGS['embeds']]
    all_sarsd = {key: [] for key in keys}
    all_dyn_vals = defaultdict(lambda: [])
    new_rollouts = []

    for ridx, rollout in enumerate(rollouts):
        if len(rollout.rewards) == 1:  # this happens if the env is done right on a horizon boundary, so we skip these rollouts
            # TODO: log how often this happens
            continue

        obses = obs_vectorizer.to_vecs(rollout.observations)
        sarsd = {}
        sarsd['s'] = obses[:-1]
        sarsd['a'] = np.concatenate([actvec(mo['actions']) for mo in rollout.model_outs])[:-1]
        sarsd['s_next'] = obses[1:]
        sarsd['r'] = rollout.rewards[:len(sarsd['s'])]
        sarsd['d'] = np.zeros_like(sarsd['r'])

        if dyn is not None:
            dyn_vals = dyn.step(subdict(sarsd, ['s','a','s_next']))
            dyn_vals['reward_phi_s'] = dyn_vals['phi_s']
            dyn_vals['reward_phi_s_next'] = dyn_vals['phi_s_next']
            dyn_vals['reward_phi_g'] = dyn_vals['phi_g']

            append(all_dyn_vals, copy.deepcopy(dyn_vals))

            if FLAGS['use_embed']:
                for key in dyn_vals:
                    if 'phi' in key and 'pred' not in key:
                        sarsd[key+'_pi'] = dyn_vals[key]
                if value_dyn is not None:
                    value_dyn_vals = value_dyn.step(subdict(sarsd, ['s','a','s_next']))
                    value_dyn_vals['reward_phi_s'] = value_dyn_vals['phi_s']
                    value_dyn_vals['reward_phi_s_next'] = value_dyn_vals['phi_s_next']
                    value_dyn_vals['reward_phi_g'] = value_dyn_vals['phi_g']

                    append(all_dyn_vals, copy.deepcopy(prefix_vals('value', value_dyn_vals)))
                    for key in value_dyn_vals:
                        if 'phi' in key and 'pred' not in key:
                            sarsd[key+'_vf'] = value_dyn_vals[key]

        append(all_sarsd, copy.deepcopy(sarsd))
        new_rollouts.append(rollout)

    if dyn is not None:
        swapped_rollouts, swapped_sarsd, swapped_dyn_vals, extra_info = swap_rewards(new_rollouts, all_sarsd, all_dyn_vals, obs_vectorizer, FLAGS)
        #import ipdb; ipdb.set_trace() # TODO: pop s and s_next to save space
        # TODO: add assertions everywhere to ensure things are same length
        return swapped_rollouts, swapped_sarsd, swapped_dyn_vals, extra_info
    else:
        return new_rollouts, all_sarsd, all_dyn_vals, {}

def herify(sarsd, dyn_vals, obs_vectorizer, FLAGS):
    """return new sarsd's that came from HER

    NOTE: this requires that sarsd not be flattened yet. a bit hacky
    
    """
    her_sarsd = []
    legit_reached_goals = []
    no_move_reach = []
    moves = []

    for ridx in range(len(sarsd['s'])):
        all_s = sarsd['s'][ridx]
        all_s_next = sarsd['s_next'][ridx]
        phi_s_next = dyn_vals['phi_s_next'][ridx] 
        if FLAGS['aac']:
            value_phi_s_next = dyn_vals['value_phi_s_next'][ridx] 

        # We are kind of keeping this in format of rollouts
        for k in range(FLAGS['her_k']):
            new_sarsd = {}
            new_sarsd['d'] = np.zeros_like(sarsd['r'][ridx])
            new_sarsd['r'] = np.zeros_like(sarsd['r'][ridx]) 
            num_tidx = len(new_sarsd['r'])
            new_sarsd['r'] += FLAGS['old_weight'] * sarsd['og_reward'][ridx][:num_tidx]

            new_s = obs_vectorizer.to_np_dict(copy.deepcopy(all_s))
            new_s_next = obs_vectorizer.to_np_dict(copy.deepcopy(all_s_next))

            next_obs = obs_vectorizer.to_np_dict(all_s_next)
            next_array = next_obs['array']
            if FLAGS['use_image'] and not FLAGS['use_embed']:
                next_image = next_obs['image']

            if FLAGS['use_embed']:
                new_sarsd['phi_s_pi'] = sarsd['phi_s_pi'][ridx].copy()
                new_sarsd['phi_s_next_pi'] = sarsd['phi_s_next_pi'][ridx].copy()
                new_sarsd['phi_g_pi'] = np.full_like(new_sarsd['phi_s_pi'], np.nan)
                if FLAGS['aac']:
                    new_sarsd['phi_s_vf'] = sarsd['phi_s_vf'][ridx].copy()
                    new_sarsd['phi_s_next_vf'] = sarsd['phi_s_next_vf'][ridx].copy()
                    new_sarsd['phi_g_vf'] = np.full_like(new_sarsd['phi_s_vf'], np.nan)

            for tidx in range(num_tidx):
                # sample random new future
                new_idx = np.random.randint(tidx, num_tidx)

                new_goal_array = next_array[new_idx].copy()
                new_s['goal_array'][tidx] = new_goal_array
                new_s_next['goal_array'][tidx] = new_goal_array

                # Replace observations
                if FLAGS['use_embed']:
                    new_sarsd['phi_g_pi'][tidx] = phi_s_next[new_idx].copy()
                    if FLAGS['aac']: 
                        new_sarsd['phi_g_vf'][tidx] = value_phi_s_next[new_idx].copy()

                if FLAGS['use_image'] and not FLAGS['use_embed']:
                    new_goal_image = next_image[new_idx].copy()
                    new_s['goal_image'][tidx] = new_goal_image
                    new_s_next['goal_image'][tidx] = new_goal_image

                # Compute replaced rewards
                moved = True
                if FLAGS['penalize_stasis']:
                    # Penalize the agent if it does not move
                    curr_s, curr_s_next = new_s['array'][tidx], new_s_next['array'][tidx]
                    deltas = (curr_s_next - curr_s)
                    sum_deltas = np.sum(np.abs(deltas))
                    moved = sum_deltas >= FLAGS['stasis_threshold']
                    if moved:
                        new_sarsd['r'][tidx] += FLAGS['stasis_rew']
                    else:
                        new_sarsd['r'][tidx] += -FLAGS['stasis_rew']
                moves.append(1 if moved else 0)

                # Reward is sparse random new goal in future
                if FLAGS['aac']:
                    # compute goal using value (GN) because it is probably a bit better
                    #dgoal = cos_dist(value_phi_s_next[tidx], value_phi_s_next[new_idx])
                    dgoal = cos_dist(dyn_vals['value_reward_phi_s_next'][ridx][tidx], dyn_vals['value_reward_phi_s_next'][ridx][new_idx])
                else:
                    #goal = cos_dist(phi_s_next[tidx], phi_s_next[new_idx])
                    dgoal = cos_dist(dyn_vals['reward_phi_s_next'][ridx][tidx], dyn_vals['reward_phi_s_next'][ridx][new_idx])

                if dgoal < FLAGS['goal_threshold']:
                    if moved:
                        new_sarsd['r'][tidx] += 1.0
                        legit_reached_goals.append(1)
                        no_move_reach.append(0)
                    else:
                        no_move_reach.append(1)
                        legit_reached_goals.append(0)
                    # DONE
                    new_sarsd['d'][tidx] = 1.0  # done if we reach goal
                else:
                    new_sarsd['r'][tidx] += -1.0 # else -1
                    no_move_reach.append(0)
                    legit_reached_goals.append(0)


            # combine new stuff
            if not FLAGS['use_embed']:
                new_sarsd['s'] = obs_vectorizer.to_vecs([ns for ns in zip(*new_s.values())])
                new_sarsd['s_next'] = obs_vectorizer.to_vecs([ns_next for ns_next in zip(*new_s_next.values())])
            new_sarsd['a'] = sarsd['a'][ridx]
            her_sarsd.append(new_sarsd)

            # TODO: add all new stuff to a sarsd or rollout
    out_sarsd = {}
    for key in her_sarsd[0].keys():
        out_sarsd[key] = np.concatenate([hs[key] for hs in her_sarsd])
    extra_info = {'her_reach_frac': np.mean(legit_reached_goals), 'her_nomove_reach_frac': np.mean(no_move_reach), 'her_move_frac': np.mean(moves)}

    return out_sarsd, extra_info


def varscale_rewards(rollouts, mean_rms, reward_scale=1.0):
    new_rollouts = []
    for i in range(len(rollouts)):
        shallow_new_rollout = rollouts[i].copy()
        shallow_new_rollout.rewards /= np.sqrt(mean_rms.var)
        shallow_new_rollout.rewards *= reward_scale
        new_rollouts.append(shallow_new_rollout)
    return new_rollouts

def batches(rollouts, obs_vectorizer, batch_size):
    obses, rollout_idxs, timestep_idxs = frames_from_rollouts(rollouts)
    for mini_indices in mini_batches([1] * len(obses), batch_size):
        sub_obses = [obses[i] for i in mini_indices]
        yield {
            'rollout_idxs': np.take(rollout_idxs, mini_indices),
            'timestep_idxs': np.take(timestep_idxs, mini_indices),
            'obses': obs_vectorizer.to_vecs(sub_obses)
        }

def frames_from_rollouts(rollouts):
    """
    Flatten out the rollouts and produce a list of
    observations, rollout indices, and timestep indices.

    Does not include trailing observations for truncated
    rollouts.

    For example, [[obs1, obs2], [obs3, obs4, obs5]] would
    become ([obs1, obs2, ..., obs5], [0, 0, 1, 1, 1],
    [0, 1, 0, 1, 2])
    """
    all_obs = []
    rollout_indices = []
    timestep_indices = []
    for rollout_idx, rollout in enumerate(rollouts):
        for timestep_idx, obs in enumerate(rollout.step_observations):
            all_obs.append(obs)
            rollout_indices.append(rollout_idx)
            timestep_indices.append(timestep_idx)
    return all_obs, rollout_indices, timestep_indices


def select_from_batch(advs, batch):
    """
    Take a rollout-shaped list of lists and select the
    indices from the mini-batch.
    """
    indices = zip(batch['rollout_idxs'], batch['timestep_idxs'])
    return [advs[x][y] for x, y in indices]


def select_model_out_from_batch(key, rollouts, batch):
    """
    Select a model_outs key corresponding to the indices
    from the mini-batch.
    """
    vals = [[m[key][0] for m in r.model_outs] for r in rollouts]
    return select_from_batch(vals, batch)


def mini_batches(size_per_index, batch_size=None):
    """
    Generate mini-batches of size batch_size.

    The size_per_index list is the size of each batch
    element.
    Batches are generated such that the sum of the sizes
    of the batch elements is at least batch_size.
    """
    if batch_size is None or sum(size_per_index) <= batch_size:
        while True:
            yield list(range(len(size_per_index)))
    cur_indices = []
    cur_size = 0
    for idx in _infinite_random_shuffle(len(size_per_index)):
        cur_indices.append(idx)
        cur_size += size_per_index[idx]
        if cur_size >= batch_size:
            yield cur_indices
            cur_indices = []
            cur_size = 0


def _infinite_random_shuffle(num_elements):
    """
    Continually permute the elements and yield all of the
    permuted indices.
    """
    while True:
        for elem in np.random.permutation(num_elements):
            yield elem


def convert_to_dict_dist(space):
    #return gym_space_distribution(Tuple([space[key] for key in space]))
    keys = list(sorted(space.keys()))
    return DictDistribution({key: gym_space_distribution(space[key]) for key in space})

class DictDistribution(TupleDistribution):
    """
    A distribution that consists of a dictionary of sub-distributions.
    """

    def __init__(self, dists, to_sample=lambda x: x):
        self.dict_dists = dists
        self.to_sample = to_sample
        super().__init__(dists.values(), to_sample=to_sample)

    @property
    def dout_shape(self):
        return {key: self.dict_dists[key].out_shape for key in self.dict_dists}

    def to_combined(self, dict):
        combined = tuple(dict.values())
        return combined

    def to_np_dict(self, combined):
        return self._to_dict(combined, np)

    def to_tf_dict(self, combined):
        return self._to_dict(combined, tf)

    def idxs(self):
        dists = self.dict_dists
        idxs = list(itertools.accumulate([0] + [np.prod(dists[key].out_shape) for key in sorted(dists.keys())]))
        return idxs

    def _to_dict(self, combined, rs):
        dists = self.dict_dists
        idxs = list(itertools.accumulate([0] + [np.prod(dists[key].out_shape) for key in sorted(dists.keys())]))
        dict = {key: combined[:,idxs[i]:idxs[i+1]] for i, key in enumerate(sorted(dists.keys()))}
        dict = {key: rs.reshape(dict[key], (-1,)+dists[key].out_shape) for key in sorted(dict.keys())}
        return dict

    def dto_vecs(self, space_elements):
        return {key: self.dict_dists[key].to_vecs(list(space_elements[key])) for key in self.dict_dists}

    @property
    def dparam_shape(self):
        return {key: self.dict_dists[key].param_shape for key in self.dict_dists}

    def dsample(self, param_batch):
        return {key: self.dict_dists[key].sample(param_batch[key]) for key in self.dict_dists}

    def dmode(self, param_batch):
        return {key: self.dict_dists[key].mode(param_batch[key]) for key in self.dict_dists}

    # TODO: we may need to tf_add_n some of these to match other interfaces
    def dlog_prob(self, param_batch, sample_vecs):
        return {self.dict_dists[key].log_prob(param_batch[key], sample_vecs[key]) for key in self.dict_dists}

    def dentropy(self, param_batch):
        return {self.dict_dists[key].entropy(param_batch[key]) for key in self.dict_dists}

    def dkl_divergence(self, param_batch_1, param_batch_2):
        return {self.dict_dists[key].kl_divergence(param_batch_1[key], param_batch_2[key]) for key in self.dict_dists}



def py_gif_summary(tag, paths, max_outputs):
    """Outputs a `Summary` protocol buffer with gif animations.
    Args:
        tag: Name of the summary.
    Returns:
        The serialized `Summary` protocol buffer.
    Raises:
        ValueError: If `images` is not a 5-D `uint8` array with 1 or 3 channels.
    """
    is_bytes = isinstance(tag, bytes)
    if is_bytes:
        tag = tag.decode("utf-8")

    summ = tf.Summary()
    for i in range(min(len(paths), max_outputs)):
        image_summ = tf.Summary.Image()
        image_summ.height = 480
        image_summ.width = 640
        image_summ.colorspace = 3

        #image_summ.encoded_image_string = PIL.Image.open(paths[i].decode()).tobytes()
        # TODO: fix gif offsets
        # http://www.onicos.com/staff/iz/formats/gif.html
        with open(paths[i].decode(), 'rb') as f:
            image_summ.encoded_image_string = f.read()

        if max_outputs == 1:
            summ_tag = "{}/gif".format(tag)
        else:
            summ_tag = "{}/gif/{}".format(tag, i)
        summ.value.add(tag=summ_tag, image=image_summ)
    summ_str = summ.SerializeToString()
    return summ_str


def gif_summary_v2(name, paths, max_outputs=3, family=None, step=None):

    def py_gif_event(step, tag, paths, max_outputs):
        summary = py_gif_summary(tag, paths, max_outputs)

        if isinstance(summary, bytes):
            summ = summary_pb2.Summary()
            summ.ParseFromString(summary)
            summary = summ

        event = event_pb2.Event(summary=summary)
        event.wall_time = time.time()
        event.step = step
        event_pb = event.SerializeToString()
        return event_pb

    def function(tag, scope):
        # Note the identity to move the paths to the CPU.
        event = tf.py_func(
                py_gif_event,
                [_choose_step(step), tag, tf.identity(paths), max_outputs],
                tf.string,
                stateful=False)
        return summary_ops_v2.import_event(event, name=scope)

    return summary_ops_v2.summary_writer_function(
            name, paths, function, family=family)


def _choose_step(step):
    if step is None:
        return tf.train.get_or_create_global_step()
    if not isinstance(step, tf.Tensor):
        return tf.convert_to_tensor(step, tf.int64)
    return step


def gif_summary(name, paths, max_outputs, collections=None, family=None):
  """Outputs a `Summary` protocol buffer with gif animations.
  Args:
    name: Name of the summary.
    tensor: A 5-D `uint8` `Tensor` of shape `[batch_size, time, height, width,
      channels]` where `channels` is 1 or 3.
    max_outputs: Max number of batch elements to generate gifs for.
    collections: Optional list of tf.GraphKeys.  The collections to add the
      summary to.  Defaults to [tf.GraphKeys.SUMMARIES]
    family: Optional; if provided, used as the prefix of the summary tag name,
      which controls the tab name used for display on Tensorboard.
  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  """
  paths = tf.convert_to_tensor(paths)
  if summary_op_util.skip_summary():
    return tf.constant("")
  with summary_op_util.summary_scope(
      name, family, values=[paths]) as (tag, scope):
    val = tf.py_func(
        py_gif_summary,
        [tag, paths, max_outputs],
        tf.string,
        stateful=False,
        name=scope)
    summary_op_util.collect(val, collections, [tf.GraphKeys.SUMMARIES])
  return val


class Timer:     
    def __init__(self, message, FLAGS):
        self.FLAGS = FLAGS
        self.message = message
    def __enter__(self): 
        self.start = time.time() 
        return self 
 
    def __exit__(self, *args): 
        self.end = time.time() 
        self.interval = self.end - self.start 

        if self.FLAGS['debug']:
            self.display()

    def display(self):
        print()
        print(self.message, self.interval, "s") 
        print()



def plot_rollout(obs_vectorizer, action_dist, FLAGS, itr, rollouts, dgoals=None, simis=None):
    def get_state(rollout): return obs_vectorizer.to_np_dict(obs_vectorizer.to_vecs(rollout.observations))
    # TODO: clean this up

    def update_plot(i, scat1, scat2, text, arrow, state, actions, dgoal=None, simi=None, seed=0):
        #input()

        action_d = action_dist.to_np_dict(action_dist.to_vecs(actions))
        if FLAGS['discrete']:
            for key in action_d: 
                action_d[key] = FLAGS['ACTS'][key][action_d[key]][i]
        else:
            for key in action_d: 
                if key not in ['x', 'y']:
                    action_d[key] = unmap_continuous(key, np.squeeze(action_d[key][i]), FLAGS)
                else:
                    action_d[key] = np.squeeze(action_d[key][i])

        #pp['x'] /= 0.5*TABLES[FLAGS['default_table']]['wood'][0]
        #pp['y'] /= 0.5*TABLES[FLAGS['default_table']]['wood'][1]
        #print(i, pp)

        s = state['array'][i,:,:2] 
        scat1.set_offsets(s)

        dx = action_d['x'] + (np.cos(action_d['yaw'])*action_d['dist'] / (0.5*TABLES[FLAGS['default_table']]['wood'][0]))
        dy = action_d['y'] + (np.sin(action_d['yaw'])*action_d['dist'] / (0.5*TABLES[FLAGS['default_table']]['wood'][1]))
        arrow.set_xy([[action_d['x'],action_d['y']],[dx, dy]])

        if dgoals is not None and simis is not None:
            g = state['goal_array'][i,:,:2] 
            scat2.set_offsets(g)
            text.set_text('env {0} :  step {1}  dgoal = {2:.4f} simi = {3:.4f}'.format(seed, i, dgoal[i], simi[i]))
            return scat1, scat2, text, arrow
        else:
            text.set_text('env {0} :  step {1}'.format(seed, i))
            return scat1, text, arrow



    # TODO: set these randomly
    paths = []

    for ridx in range(len(rollouts)):
        rollout = rollouts[ridx]

        state = get_state(rollout)
        actions = [mo['actions'][0] for mo in rollout.model_outs]
        if dgoals is not None and simis is not None:
            dgoal = dgoals[ridx]
            simi = simis[ridx]
        else:
            dgoal = None
            simi = None
        seed = rollout.infos[0]['seed']

        fig = plt.figure()
        numframes = len(actions)
        numpoints = FLAGS['num_objects']
        if FLAGS['goal_conditioned']:
            scat2 = plt.scatter(np.linspace(-1.0, 1.0, numpoints), np.linspace(-1.0, 1.0, numpoints), s=100.0, label='goal', alpha=0.4)

        scat1 = plt.scatter(np.linspace(-1.0, 1.0, numpoints), np.linspace(-1.0, 1.0, numpoints), s=100.0, label='state')
        arrow = plt.arrow(0, 0, 0.5, 0.5, shape='right', lw=5, length_includes_head=True, head_length=0.5, head_width=0.5, overhang=0.1)
        text = plt.text(-1.0, 1.0, 'text')
        plt.legend(loc='upper right')

        if FLAGS['goal_conditioned']:
            ani = animation.FuncAnimation(fig, update_plot, frames=range(numframes), fargs=(scat1, scat2, text, arrow, state, actions, dgoal, simi, seed))
        else:
            ani = animation.FuncAnimation(fig, update_plot, frames=range(numframes), fargs=(scat1, None, text, arrow, state, actions, seed))

        name = 'rollout_animation_{}-{}-{}.gif'.format(itr, seed, ridx)
        path = os.path.join(FLAGS['plot_path'], name)
        ani.save(path, fps=2, writer='imagemagick')
        #ani.save('rollout_animation_{}-{}.mp4'.format(itr, seed), fps=2, extra_args=['-vcodec', 'libx264'])
        plt.close()
        paths.append(path)

    return paths


def clipped_objective(new_log_probs, old_log_probs, advs, eps):
    """
    Compute the component-wise clipped PPO objective.
    """

    prob_ratio = tf.exp(new_log_probs - old_log_probs)
    clipped_ratio = tf.clip_by_value(prob_ratio, 1-eps, 1+eps)
    objective = tf.minimum(advs*clipped_ratio, advs*prob_ratio)

    clip_frac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(prob_ratio - 1.0), eps)))
    return objective, clip_frac


def clipped_value_loss(new_values, old_values, targets, eps):
    """
    Compute the component-wise clipped value objective.

    (same motivation as why to clip policy.  don't change too much, but allow us to fix mistakes in wrong direction)
     https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl/50663200#50663200)
    """
    diffs = new_values - old_values
    clipped_values = old_values + tf.clip_by_value(diffs, -eps, eps)

    clip_loss = tf.square(clipped_values - targets)
    unclip_loss = tf.square(old_values - targets)
    value_loss = tf.reduce_mean(tf.maximum(clip_loss, unclip_loss))

    clip_frac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(diffs), eps)))
    return value_loss, clip_frac
