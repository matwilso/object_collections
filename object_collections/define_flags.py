"""
Command-line argument parsing.
"""
import datetime
import itertools
import os
import subprocess
import sys
import copy
import functools
import matplotlib

from gym import error, spaces
import numpy as np
import tensorflow as tf
import yaml
from tensorflow.python.platform import flags
try:
    import pyperclip
    clipboard = True
except:
    clipboard = False
    print('no clipboard available')

from object_collections.envs.utils.table import TABLES

def R(min, max):
    if min <= max:
        return np.array([min, max])
    else:
        print('WARNING: min {} was greater than max {}'.format(min, max))
        return np.array([max, min])

INITS = {'zeros': tf.keras.initializers.zeros(dtype=tf.float32), 'he': tf.keras.initializers.he_normal(), 'glorot': tf.keras.initializers.glorot_uniform(), 'ortho': tf.keras.initializers.orthogonal(gain=np.sqrt(2), dtype=np.float32),'none': None}
REV_INITS = {v.__class__:k for k,v in INITS.items()}
ACTIVS = {'relu': tf.nn.relu, 'tanh': tf.tanh, 'leaky_relu': tf.nn.leaky_relu}
REV_ACTIVS = {v:k for k,v in ACTIVS.items()}
def subdict(dict, keys): return {key: dict[key] for key in keys}

FLAGS = flags.FLAGS

flag_filename = 'flags.yaml'
def dump_flags(FLAGS):
    """Write flags to file"""
    dumping_flags = copy.deepcopy(FLAGS)
    dumping_flags['activ'] = REV_ACTIVS[dumping_flags['activ']]
    if dumping_flags['init'] is not None:
        dumping_flags['init'] = REV_INITS[dumping_flags['init']['w'].__class__]
    else:
        dumping_flags['init'] = 'none'
    dumping_flags.pop('ACTIVS')
    dumping_flags.pop('INITS')
    with open(os.path.join(FLAGS['log_path'], flag_filename), 'w') as f:
        yaml.dump(dumping_flags, f)

def load_flags(FLAGS):
    with open(os.path.join(FLAGS['load_flag_path'], flag_filename), 'r') as f:
        loaded_flags = yaml.load(f)
    if FLAGS['dont_load'] != '':
        for flag in FLAGS['dont_load'].split(','):
            loaded_flags[flag] = FLAGS[flag]

    return loaded_flags

# GENERAL stuff
flags.DEFINE_float('lr', 3e-4, help='learning rate')
flags.DEFINE_integer('bs', 512, help='batch size')
flags.DEFINE_string('init', 'none', help='ortho,glorot,he,TODO:snt')
flags.DEFINE_string('activ', 'tanh', help='activation')
flags.DEFINE_bool('clipboard', True, help='save path to clipboard for easy pasting')
flags.DEFINE_bool('inference', False, help='flag to flip at inference time')
flags.DEFINE_bool('clip_gradients', True, help='')
flags.DEFINE_integer('num_epochs', 15, help='')
flags.DEFINE_float('beta1', 0.9, help='')
flags.DEFINE_bool('norm_grads', False, help='')
flags.DEFINE_bool('domrand', True, help='apply domain randomization')
flags.DEFINE_bool('baxter', True, help='use baxter robot (vs. kuka)')
flags.DEFINE_bool('enforce_prior', False, help='enforce prior on autoencoder to make it a vae (found vae did not work well)')

# ENV stuff 
flags.DEFINE_integer('nsubsteps', 1, help='in env')
flags.DEFINE_bool('display_data', False, help='plot data for debugging env')
flags.DEFINE_bool('mj_gpu', True, help='use gpu in mujoco')
flags.DEFINE_bool('render', False, help='')
flags.DEFINE_bool('play', False, help='test a trained model')
flags.DEFINE_bool('test', False, help='')
flags.DEFINE_bool('scripted_test', False, help='run scripted policy to test and debug env')
flags.DEFINE_bool('random_policy', False, help='')
flags.DEFINE_integer('image_height', 84, help='')
flags.DEFINE_integer('image_width', 84, help='')
#flags.DEFINE_float('act_range', 0.4, help='')
flags.DEFINE_integer('num_objects', 10, help='')
flags.DEFINE_integer('seed', 0, help='')
flags.DEFINE_bool('use_quat', False, help='use quaternion in the state.')
# actions
flags.DEFINE_bool('discrete', False, help='')
flags.DEFINE_integer('act_x_n', 32, help='')
flags.DEFINE_integer('act_y_n', 32, help='')
flags.DEFINE_integer('act_yaw_n', 8, help='')
flags.DEFINE_bool('use_dist', True, help='')
flags.DEFINE_integer('act_dist_n', 2, help='')
flags.DEFINE_float('max_dx', 0.3, help='TODO: rename to max push dist')
# reward (for single)
flags.DEFINE_float('reward_scale', 1.0, help='')
flags.DEFINE_bool('goal_conditioned', False, help='if agent policy network expects to receive a goal.  False during data collection and SL training.  True during RL')
flags.DEFINE_float('goal_threshold', 0.05, help='epsilon in the paper')
flags.DEFINE_string('reset_mode', 'uniform,cluster', help='mode to use for resetting')
flags.DEFINE_bool('grad_summaries', False, help='debugging tool')
flags.DEFINE_string('cnn_gn', 'gnn', help='what type of model to use for training')
flags.DEFINE_string('goal_dyn', '', help='')
flags.DEFINE_bool('use_bn', True, help='')
flags.DEFINE_bool('cnn_train_match', False, help='')
flags.DEFINE_bool('dyn_prob', False, help='')
flags.DEFINE_bool('dyn_coadapt', False, help='')
flags.DEFINE_bool('is_training', True, help='Used for batch norm to switch to inference mode. True during SL and False after that')
flags.DEFINE_bool('view_invisiwall', False, help='')
flags.DEFINE_bool('mixed_script', False, help='')
flags.DEFINE_bool('mild_canonical', False, help='')

# RL stuff
flags.DEFINE_string('rl_model', 'gformer', help='')
flags.DEFINE_integer('num_envs', 8, help='')
flags.DEFINE_integer('horizon', 64, help='horizon for training')
flags.DEFINE_integer('max_episode_steps', 50, help='')
flags.DEFINE_string('value_network', 'shared', help='shared means they share a base.  copy means they each have their own independent networks. I suspect copy will be more stable, but computationally expensive.')
flags.DEFINE_bool('threading', True, help='')
flags.DEFINE_bool('penalize_invisiwall', True, help='')
flags.DEFINE_bool('penalize_stasis', True, help='')
flags.DEFINE_bool('only_stasis', False, help='')
flags.DEFINE_float('stasis_threshold', 0.01, help='')
flags.DEFINE_float('stasis_rew', 0.1, help='')
flags.DEFINE_string('agent', 'sac', help='')
flags.DEFINE_string('rollout_data_path', 'data/rollouts', help='')
flags.DEFINE_integer('num_files', 0, help='')
flags.DEFINE_bool('dump_rollouts', False, help='')
flags.DEFINE_bool('run_rl_optim', True, help='')
flags.DEFINE_bool('debug', False, help='')
flags.DEFINE_bool('check_data', True, help='')
flags.DEFINE_bool('load_rollouts', True, help='')
flags.DEFINE_bool('shuffle', True, help='')
flags.DEFINE_bool('shuffle_files', True, help='')
flags.DEFINE_float('old_weight', 1.0, help='')
flags.DEFINE_bool('use_image', True, help='')
flags.DEFINE_bool('use_canonical', False, help='')
flags.DEFINE_bool('no_backward_penalty', False, help='')
flags.DEFINE_integer('agent_burn_in', 0, help='')
flags.DEFINE_integer('dyn_warm_start', 1, help='')
flags.DEFINE_bool('share_dyn', True, help='')

flags.DEFINE_float('polyak', 0.995, help='')
flags.DEFINE_float('replay_size', 1e6, help='')
flags.DEFINE_float('min_replay_size', 2000, help='')

flags.DEFINE_bool('use_her', True, help='')
flags.DEFINE_integer('her_k', 8, help='number of extra virtual goals to add to the replay buffer per step')
flags.DEFINE_integer('explore_anneal', 4000, help='number of iterations before fully annealing using the scripted policy')
flags.DEFINE_float('explore_frac', 0.1, help='percentage of time that a scripted policy is used after it has been annealed from 1')
flags.DEFINE_float('sac_alpha', 0.1, help='')
flags.DEFINE_integer('optim_steps', 50, help='')
flags.DEFINE_bool('use_embed', True, help='')
flags.DEFINE_bool('aac', True, help='asymmetric actor critic')
flags.DEFINE_bool('value_goal', False, help='use value encoder to always embed the goal')

flags.DEFINE_float('dyn_lr', 3e-4, help='')
flags.DEFINE_bool('mdn_aux', True, help='')
flags.DEFINE_integer('dyn_rep_size', 128, help='')
flags.DEFINE_float('fwd_coeff', 1.0, help='')
flags.DEFINE_float('phi_noise', 0.0, help='amount of noise added to the latent space before decoding.  helps in data augmentation and robustifying i guess.')
flags.DEFINE_bool('abs_noise', False, help='')
flags.DEFINE_float('dyn_weight', 1.0, help='')
flags.DEFINE_float('mdn_weight', 1.0, help='')

# TRAINING stuff
flags.DEFINE_string('mlog_dir', 'logs/', help='')
flags.DEFINE_string('eval_image_path', 'data/real', help='')
flags.DEFINE_string('plot_path', '', help='')
flags.DEFINE_integer('eval_n', 100, help='')
flags.DEFINE_integer('save_n', 1000, help='')
flags.DEFINE_float('total_n', 5e5, help='')
flags.DEFINE_string('save_path', '', help='')
flags.DEFINE_string('load_path', '', help='')
flags.DEFINE_bool('augment_data', True, help='')

# TRANSFORMER stuff
# there are different variables for cnn and gnn, but usually they are the same. it was just because i had to frankenstein load some models once
flags.DEFINE_integer('cnn_hidden_size', 128, help='')
flags.DEFINE_string('cnn_activ', 'tanh', help='')
flags.DEFINE_string('gn_activ', 'tanh', help='')
flags.DEFINE_bool('cnn_gn_coord', True, help='')
flags.DEFINE_integer('num_mlp_layers', 3, help='')
flags.DEFINE_integer('mlp_hidden_size', 256, help='')


flags.DEFINE_integer('tf_hidden_size', 128, help='')
flags.DEFINE_integer('tf_num_heads', 8, help='')
flags.DEFINE_bool('tf_gate', True, help='')
flags.DEFINE_string('tf_reducer', 'sum', help='')

# CNN stuff
flags.DEFINE_string('conv_flatten', 'flatten', help='flatten, ssam, linear, no')
flags.DEFINE_integer('conv_filters', 64, help='')
flags.DEFINE_bool('deeper_net', True, help='')

# VAE stuff
flags.DEFINE_integer('num_vae_samples', 16, help='')
flags.DEFINE_integer('vae_k', 100, help='num components')
flags.DEFINE_integer('vae_z_size', 128, help='')
flags.DEFINE_float('vae_b', 1.0, help='b-vae (how much to extra weight KL div')

# MDN stuff
flags.DEFINE_integer('mdn_k', 25, help='num components')
flags.DEFINE_string('kl_mode', 'combined', help='')

# MISC stuff
flags.DEFINE_string('log_format', 'stdout,log,csv,tensorboard', help='')
flags.DEFINE_string('f', '', help="Just so jupyter doesn't get mad when importing this")
flags.DEFINE_string('suffix', '', help='')
flags.DEFINE_bool('date', True, help='')
flags.DEFINE_string('load_flag_path', '', help='')
flags.DEFINE_string('dont_load', '', help='')
flags.DEFINE_string('mpl_backend', 'Agg', help='')

FLAGS.f
matplotlib.use(FLAGS.mpl_backend)
def _misc_hp_str(FLAGS):
    hp_str = ''
    hp_str += '{}'.format(FLAGS['date'])
    hp_str += '-{}'.format(FLAGS['suffix']) if FLAGS['suffix'] else ''
    return hp_str


# MAKE HYPER PARAMETER STRING FOR LOGGING
def _make_hp_str(FLAGS):
    hp_str = ''
    hp_str += FLAGS['exe_name'] + '/'
    hp_str += _misc_hp_str(FLAGS)
    return hp_str

# Convert flags to a dictionary
FLAGS = {key: val.value for key, val in FLAGS._flags().items()}
FLAGS['full_cmd'] = ' '.join(sys.argv)
FLAGS['exe_name'] = sys.argv[0]
if FLAGS['exe_name'][:2] == './':
    FLAGS['exe_name'] = FLAGS['exe_name'][2:]

def post_load_flags(FLAGS):
    # PRE HP STR
    if 'colab' in os.environ:
        FLAGS['exe_name'] = 'colab_main.py'

    if FLAGS['baxter']:
        FLAGS['filepath'] = './object_collections/envs/assets/xmls/baxter/baxter.xml'
        FLAGS['default_table'] = 'folding'
    else:
        FLAGS['filepath'] = './object_collections/envs/assets/xmls/kuka/lbr4_reflex.xml'
        FLAGS['default_table'] = 'small'
    
    #if FLAGS['play']:
        #FLAGS['mj_gpu'] = True
    
    FLAGS['total_n'] = int(FLAGS['total_n'])
    FLAGS['replay_size'] = int(FLAGS['replay_size'])
    FLAGS['min_replay_size'] = int(FLAGS['min_replay_size'])
    FLAGS['date'] = datetime.datetime.today().strftime('%m-%d-%H-%M-%S') if FLAGS['date'] else '0'
    FLAGS['max_episode_steps'] = None if FLAGS['max_episode_steps'] < 1 else FLAGS['max_episode_steps']
    
    _data_path = FLAGS['rollout_data_path']
    os.makedirs(_data_path, exist_ok=True)
    _files = os.listdir(_data_path)
    _files = [f for f in _files if 'tfrecord' in f]
    FLAGS['filenames'] = list(sorted(map(lambda x: os.path.join(_data_path, x), _files)))
    if FLAGS['num_files'] not in [0, None, -1]:
        FLAGS['filenames'] = FLAGS['filenames'][:FLAGS['num_files']]

    metadata_path = os.path.join(_data_path, 'metadata.yaml')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as _f:
            FLAGS['rollouts_metadata'] = yaml.load(_f)
    
    FLAGS['hp_str'] = _make_hp_str(FLAGS)

    # POST HP STR
    if isinstance(FLAGS['reset_mode'], str):
        FLAGS['reset_mode'] = FLAGS['reset_mode'].split(',')
    FLAGS['INITS'] = INITS
    FLAGS['ACTIVS'] = ACTIVS
    if isinstance(FLAGS['init'], str):
        FLAGS['init'] = {'w': INITS[FLAGS['init']]}
        if FLAGS['init']['w'] is None:
            FLAGS['init'] = None
        FLAGS['activ'] = ACTIVS[FLAGS['activ']]
    
    pref = ['phi_s_', 'phi_s_next_', 'phi_g_']
    suff = ['pi'] 
    suff += ['vf'] if FLAGS['aac'] else []
    FLAGS['embeds'] = list(map(''.join, itertools.product(pref, suff)))
    FLAGS['DIM'] = 2
    FLAGS['mdn_flat_shape'] = FLAGS['mdn_k'] + 2 * FLAGS['DIM'] * FLAGS['mdn_k']
    FLAGS['embed_shape'] = FLAGS['dyn_rep_size']
    
    # Action space stuff so that not all code needs to have the env or know about the env
    acts = list(sorted(['x', 'y', 'yaw', 'dist']))
    if not FLAGS['use_dist']: acts.remove('dist')
    FLAGS['act_names'] = acts
    
    if FLAGS['discrete']:
        action_space = spaces.Dict({key: spaces.Discrete(FLAGS['act_{}_n'.format(key)]) for key in acts})
    else:
        # NOTE: these are all in range -1,1 so that scalings are consistent, for example if we are using TD3 noise
        action_space = spaces.Dict({key: spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32) for key in acts})
    
    FLAGS['action_space'] = action_space
    FLAGS['state_shape'] = state_shape = (FLAGS['num_objects'], 7) if FLAGS['use_quat'] else (FLAGS['num_objects'], 2)
    
    obs_dict = {
        'array': spaces.Box(low=-np.inf, high=np.inf, shape=state_shape, dtype=np.float32),
        'single': spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
    }
    if FLAGS['use_image']:
        FLAGS['image_shape'] = image_shape = (FLAGS['image_height'], FLAGS['image_width'], 3)
        obs_dict['image'] = spaces.Box(low=0, high=255, shape=image_shape, dtype=np.float32)

        if FLAGS['use_canonical'] or ('sl' in FLAGS['exe_name'] and FLAGS['rollouts_metadata']['FLAGS']['use_canonical']):
            obs_dict['canonical'] = spaces.Box(low=0, high=255, shape=image_shape, dtype=np.float32)
    
    if FLAGS['goal_conditioned']:
        obs_dict['goal_array'] = spaces.Box(low=-np.inf, high=np.inf, shape=state_shape, dtype=np.float32)
        if FLAGS['use_image']:
            obs_dict['goal_image'] = spaces.Box(low=0, high=255, shape=image_shape, dtype=np.float32)

    sorted_obs_dict = {key: obs_dict[key] for key in sorted(obs_dict.keys())}
    FLAGS['observation_space'] = spaces.Dict(sorted_obs_dict)
    
    st = 0.5*TABLES[FLAGS['default_table']]['wood']
    xrad = st[0] * 0.8
    yrad = st[1] * 0.92

    X = np.linspace(-xrad, xrad, FLAGS['act_x_n'])
    Y = np.linspace(-yrad, yrad, FLAGS['act_x_n'])
    YAW = np.linspace(-np.pi, np.pi, FLAGS['act_yaw_n'])
    DIST = np.linspace(0, FLAGS['max_dx'], FLAGS['act_dist_n']+1)
    FLAGS['ACTS'] = {'x': X, 'y': Y, 'yaw': YAW, 'dist': DIST}
    FLAGS['obj_rangex'] = 0.9 * R(-xrad, xrad)
    FLAGS['obj_rangey'] = 0.9 * R(-yrad, yrad)
    
    if 'main' in FLAGS['exe_name']:
        FLAGS['log_path'] = os.path.join(FLAGS['mlog_dir'], FLAGS['hp_str'])
        os.makedirs(FLAGS['log_path'], exist_ok=True)
        FLAGS['plot_path'] = os.path.join(FLAGS['log_path'], 'data/')
        os.makedirs(FLAGS['plot_path'], exist_ok=True)
    
    
    if 'main' in FLAGS['exe_name']:
        save_path = os.path.join(FLAGS['log_path'], 'weights/model.ckpt')
        FLAGS['save_path'] = save_path
        os.makedirs(FLAGS['save_path'], exist_ok=True)
    
    if FLAGS['load_path'] != '':
        # if the load_path is a directory, get the latest
        # else just leave the entire path (in case they wanted a specific checkpoint)
        dirname = os.path.dirname(FLAGS['load_path'])
        if abs(len(dirname) - len(FLAGS['load_path'])) <= 5:
            FLAGS['load_path'] = tf.train.latest_checkpoint(FLAGS['load_path'])
    
    assert not (FLAGS['goal_conditioned'] and FLAGS['max_episode_steps'] is None), "need to set time limit if using goals conditioned" 
    
    if 'main' in FLAGS['exe_name']:
        if FLAGS['load_flag_path'] != '':
            loaded_flags = load_flags(FLAGS) 
            pops = ['activ', 'init', 'load_path', 'load_flag_path']
            for p in pops:
                loaded_flags.pop(p)
            FLAGS.update(loaded_flags)
        else:
            print()
            print(FLAGS['log_path'])
            print()
            dump_flags(FLAGS)
    
        if not (FLAGS['test'] or FLAGS['play'] or FLAGS['scripted_test'] or FLAGS['random_policy']):
            with open('./model_log_paths.txt', 'a') as f:
                f.writelines([FLAGS['log_path'], '\n'])
            try:
                if clipboard and FLAGS['clipboard']:
                    pyperclip.copy(FLAGS['log_path'])
            except:
                pass
    return FLAGS

FLAGS = post_load_flags(FLAGS)
