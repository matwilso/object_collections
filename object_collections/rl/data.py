import copy
import os
import time
import re

import cloudpickle
import numpy as np
import PIL
import tensorflow as tf
import yaml
from tensorflow.core.framework import summary_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io.tf_record import (TFRecordCompressionType,
                                                TFRecordOptions,
                                                TFRecordWriter)
from tensorflow.python.ops import summary_ops_v2

from object_collections.sl.util import convert_nodes_to_graph_tuple
from . import util as rl_util

def prefix_vals(pre, vals): return {pre.lower() + '_' + key: vals[key] for key in vals} # add a prefix based on the name (agent or dyn)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _generate_sas_example(sas, obs_vectorizer):
    """
    This is a bit more complicated so we can use uint8 instead of float32. (4x more efficient to store)
    """
    features = {}

    # old version
    for key in sas.keys():
        features[key] = _bytes_feature(sas[key].astype(np.float32).tostring())
        
    example = tf.train.Example(
        features=tf.train.Features(
            feature=features
        )
    )
    return example

def parse_record(args, sas_shape, obs_vectorizer, FLAGS=None):
    """Decode a TF record tf.train.Example that was saved by _generate_sas_example"""
    features = {key: tf.FixedLenFeature((), tf.string) for key in ['s', 'a', 's_next']}
    parsed = tf.parse_single_example(args, features)

    two_shapes_equal = lambda x,y: tf.reduce_all(tf.equal(tf.shape(x), tf.shape(y)))

    labels = {}
    reshaped_labels = {}
    equal_ops = []
    for key in features.keys():
        labels[key] = tf.decode_raw(parsed[key], tf.float32)
        # TODO: add tf assert that this doesn't change the shape.  (This is just done so downstream knows the shape)
        reshaped_labels[key] = tf.reshape(labels[key], sas_shape[key])
        equal_ops.append(two_shapes_equal(labels[key], reshaped_labels[key]))

    equal_ops.append(two_shapes_equal(reshaped_labels['s'], labels['s_next']))

    assert_op = tf.Assert(tf.reduce_all(equal_ops), [equal_ops])
    with tf.control_dependencies([assert_op]):
        reshaped_labels = {key: tf.identity(reshaped_labels[key]) for key in reshaped_labels}
        
    return reshaped_labels

def rollout_to_tf_record(sases, obs_vectorizer, FLAGS, path=None):
    date_string = time.strftime("%Y-%m-%d-%H-%M-%S")
    path = path or FLAGS['rollout_data_path']
    filename = os.path.join(path, date_string + '.tfrecords')
    count = 0
    with TFRecordWriter(filename, options=TFRecordOptions(TFRecordCompressionType.GZIP)) as writer:
        for ridx, rollout in enumerate(sases['s']):
            for tidx in range(len(sases['s'][ridx])):
                sas = {}
                try:
                    sas['s'] = sases['s'][ridx][tidx]
                    sas['a'] = sases['a'][ridx][tidx]
                    sas['s_next'] = sases['s_next'][ridx][tidx]
                except Exception as e:
                    import ipdb; ipdb.set_trace()
                example = _generate_sas_example(copy.deepcopy(sas), obs_vectorizer)
                writer.write(example.SerializeToString())
                count += 1
    writer.close()

    # Old version. 
    sas_shape = {key: sas[key].shape for key in sas}

    dump = {'sas_shape': sas_shape, 'FLAGS': FLAGS}
    with open(os.path.join(FLAGS['rollout_data_path'], 'metadata.yaml'), 'w') as f:
        yaml.dump(dump, f)

    if FLAGS['debug']:
        print('Wrote {} SAS pairs to file'.format(count))
    return filename


def rollout_to_pickle(rollouts, sases, FLAGS):
    date_string = time.strftime("%Y-%m-%d-%H-%M-%S")
    filename = os.path.join(FLAGS['rollout_data_path'], date_string + '.cpkl')

    with open(filename, 'wb') as f:
        cloudpickle.dump({'rollouts': rollouts, 'sases': sases}, f)
    return filename

def format(args):
    pass

def augment_image(sas, sarsd):
    image = sarsd['s']['image']
    next_image = sarsd['s_next']['image']

    image = tf.image.random_contrast(image, 0.5, 1.5)
    next_image = tf.image.random_contrast(next_image, 0.5, 1.5)

    image = image + tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=0.01, dtype=tf.float32)  
    next_image = next_image + tf.random_normal(shape=tf.shape(next_image), mean=0.0, stddev=0.01, dtype=tf.float32)  

    sarsd['s']['image'] = tf.clip_by_value(image, 0.0, 1.0)
    sarsd['s_next']['image'] = tf.clip_by_value(next_image, 0.0, 1.0)

    return sas, sarsd

def split_sas(sas, obs_vectorizer, FLAGS):
    sarsd = rl_util.sarsd_to_vals(sas, obs_vectorizer, FLAGS)
    return sas, sarsd

class Dataset(object):
    """Wrapper around tf record dataset to handle sas stuff"""
    def __init__(self, obs_vectorizer, FLAGS, filenames_ph=None):
        # TODO: convert to files ph, so we can shuffle them
        if filenames_ph is None:
            filenames_ph = FLAGS['filenames']

        with open(os.path.join(FLAGS['rollout_data_path'], 'metadata.yaml'), 'r') as f:
            sas_shape = yaml.load(f)['sas_shape']

        self.dataset = tf.data.TFRecordDataset(filenames_ph, compression_type="GZIP")
        #self.dataset = self.dataset.map(normalize)
        self.dataset = self.dataset.map(lambda args: parse_record(args, sas_shape, obs_vectorizer))

        if FLAGS['shuffle']:
            self.dataset = self.dataset.shuffle(buffer_size=4096)

        self.dataset = self.dataset.batch(FLAGS['bs'])
        self.dataset = self.dataset.map(lambda sas: split_sas(sas, obs_vectorizer, FLAGS))

        if FLAGS['augment_data'] and FLAGS['use_image']:
            self.dataset = self.dataset.map(augment_image)

        self.dataset = self.dataset.prefetch(10)
        self.iterator = self.dataset.make_initializable_iterator()
        self.sas, self.sas_vals = self.iterator.get_next()
        self.state = self.sas_vals['s']
