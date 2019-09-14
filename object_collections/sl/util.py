import itertools

import numpy as np
import tensorflow as tf

from graph_nets.graphs import GraphsTuple
from .constant import ATTR_SIZE

def multi_head_loss(trainer, losses, FLAGS, var_list=None):
    # Set up multi-head loss to make sure they are about equal weight (I think)
    grads_and_vars = {key: trainer.compute_gradients(losses[key], var_list=var_list) for key in losses}

    gvs_to_gs = lambda gvs: {v: k for k, v in dict(gvs).items() if k is not None and v is not None}
    gs_to_gvs = lambda gs: [z for z in zip(gs.values(), gs.keys())]
    gs_to_name_grads = lambda gs: {k.name: v for k, v in gs.items()}
    def clip_gs(gs):
        #new_vals = [tf.clip_by_norm(v, 5.0) for v in gs.values()]
        # This seems to behave much better, though I am bit surprised by it.
        if FLAGS['norm_grads']:
            return {key: gs[key] / tf.linalg.norm(gs[key]) for key in gs}
        else:
            new_vals, _ = tf.clip_by_global_norm([v for v in gs.values()], 5.0)
            return dict(zip(gs.keys(), new_vals))

    grads = {head: gvs_to_gs(grads_and_vars[head]) for head in losses}
    if FLAGS['clip_gradients']:
        grads = {head: clip_gs(grads[head]) for head in losses}

    name_grads = {head: gs_to_name_grads(grads[head]) for head in losses}

    master_grads = {}
    all_vars = set([item for sublist in [grads[head] for head in grads.keys()] for item in sublist])

    grad_summaries = []
    eps = tf.constant(1e-9, dtype=tf.float32)
    compute_grad_mag = lambda grad: tf.reduce_mean(tf.log(tf.abs(grad) + eps))

    with tf.name_scope('grads'):
        for var in all_vars:
            master_grads[var] = 0.0
            for head in losses.keys():
                if var in grads[head]:
                    head_grad = grads[head][var]
                    if FLAGS['grad_summaries']:
                        grad_summaries.append(tf.summary.scalar(var.name+'/'+head, compute_grad_mag(head_grad)))
                    master_grads[var] += head_grad

    master_grads_and_vars = gs_to_gvs(master_grads)
    return master_grads_and_vars, grad_summaries


def convert_nodes_to_graph_tuple(nodes, FLAGS, attr_size=None):
    """
    Convert raw values into a batch of fully connected graphs in a GraphsTuple

    Args:
        nodes: A Tensor with shape (batch_size, num_nodes, attr_size)
    Returns:
        GraphsTuple where each elem in batch_size is considered a separate graph
    """
    # TODO: add support for masking out non-used idxs
    node_shape = tf.shape(nodes)

    bs, num_nodes, _ = node_shape[0], node_shape[1], node_shape[2]
    if attr_size is None:
        attr_size = 7 if FLAGS['use_quat'] else 2

    nodes = tf.reshape(nodes, tf.stack([-1, attr_size]))
    n_node = num_nodes*tf.ones(bs, dtype=tf.int32)

    # create pattern to connect every 10 nodes together in a complete graph
    # (qcount = repeating [0,1,2,3,4,5,6,7,8,...,n,0,1,....]. lcount = repeating [0,0,0,0,0,0,0,0,0,0,1,1,1,.....,n,n])
    # offset breaks up the graphs
    offset = num_nodes * tf.range(0, bs)[:,None]
    qcount = offset + tf.tile(tf.range(0, num_nodes)[None, :], [bs, num_nodes])
    lcount = tf.transpose(tf.tile(tf.range(0, num_nodes)[None, None, :], [bs, num_nodes, 1]), [0, 2, 1])
    lcount = offset + tf.reshape(lcount, [-1, num_nodes**2])

    receivers = qcount
    senders = lcount

    n_edge = bs*num_nodes
    edges = tf.zeros([n_edge,1], tf.float32)
    globals = tf.zeros(bs, tf.float32)

    graph = GraphsTuple(nodes=nodes, edges=edges, receivers=receivers, senders=senders, globals=globals, n_node=n_node, n_edge=n_edge)
    return graph


def merge_summaries(sd, id):
    summaries = []
    for key in sd.keys():
        summaries.append(tf.summary.scalar(key, sd[key]))
    for key in id.keys():
        summaries.append(tf.summary.image(key, id[key]))
    return tf.summary.merge(summaries)

def pack_images(images, rows, cols):
    """Helper utility to make a field of images."""
    shape = tf.shape(images)
    width = shape[-3]
    height = shape[-2]
    depth = shape[-1]
    images = tf.reshape(images, (-1, width, height, depth))
    batch = tf.shape(images)[0]
    rows = tf.minimum(rows, batch)
    cols = tf.minimum(batch // rows, cols)
    images = images[:rows * cols]
    images = tf.reshape(images, (rows, cols, width, height, depth))
    images = tf.transpose(images, [0, 2, 1, 3, 4])
    images = tf.reshape(images, [1, rows * width, cols * height, depth])
    return images

def image_tile_summary(name, tensor, rows=8, cols=8):
    return tf.summary.image(name, pack_images(tensor, rows, cols), max_outputs=3)

def cartesian_product(a,b):
    a, b = a[None, :, None], b[:, None, None]
    prod = tf.concat([b + tf.zeros_like(a), tf.zeros_like(b) + a], axis = 2)
    #new_shape = tf.stack([-1, tf.shape(cartesian_product)[-1]])
    #cartesian_product = tf.reshape(cartesian_product, new_shape)
    prod = tf.reshape(prod, [-1])
    return prod
