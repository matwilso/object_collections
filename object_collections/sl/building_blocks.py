import sonnet as snt
from graph_nets import modules, utils_tf, blocks
import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfd
from .util import convert_nodes_to_graph_tuple
from .constant import ATTR_SIZE
from object_collections.rl.util import flatten_mdn

softplus_inverse = lambda x: tf.log(tf.expm1(x))

"""
LEGO bricks
"""

# Base classes. 
class Module(snt.AbstractModule):
    def __init__(self, FLAGS, init=None, activ=None, name=None):
        super().__init__(name=name)
        self.FLAGS = FLAGS
        self.init = init or self.FLAGS['init']
        self.activ = activ or self.FLAGS['activ']

class Transformer(object):
    @staticmethod
    def split_heads(x, FLAGS):
        """Split x into different heads, and transpose the resulting value.
        The tensor is transposed to insure the inner dimensions hold the correct
        values during the matrix multiplication.
        Args:
            x: A tensor with shape [batch_size, length, hidden_size]
        Returns:
            A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
        """
        with tf.name_scope("split_heads"):
            # Calculate depth of last dimension after it has been split.
            depth = (FLAGS['tf_hidden_size'] // FLAGS['tf_num_heads'])

            # Split the last dimension
            x = tf.reshape(x, [-1, FLAGS['tf_num_heads'], depth])

            # Transpose the result
            return x

    @staticmethod
    def combine_heads(x, FLAGS):
        """Combine tensor that has been split.
        Args:
            x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]
        Returns:
            A tensor with shape [batch_size, length, hidden_size]
        """
        with tf.name_scope("combine_heads"):
            return tf.reshape(x, [-1, FLAGS['tf_hidden_size']])



class GraphFormer(Module, Transformer):
    """GFORMER
    This is Transformer encoder model that takes in arbitrary graphs.
        - single block
        - uses multi-head self-attention
        - residual/skip connections 
        - (optional) Wavenet actvation on the node to global aggregation output. found it helps a bit

    Wavenet: https://arxiv.org/pdf/1609.03499.pdf
    Transfomer (Attention is All You Need): https://arxiv.org/abs/1706.03762
    """ 
    def __init__(self, FLAGS, init=None, activ=None, name="GraphFormer"):
        super().__init__(FLAGS=FLAGS, init=init, activ=activ, name=name)
        self.hidden_size = self.FLAGS['tf_hidden_size']
        with self._enter_variable_scope():
            self._input_dense = snt.Linear(self.hidden_size, use_bias=False, name='input', initializers=self.init)
            self._q_dense = snt.Linear(self.hidden_size, use_bias=False, name='q', initializers=self.init)
            self._k_dense = snt.Linear(self.hidden_size, use_bias=False, name='k', initializers=self.init)
            self._v_dense = snt.Linear(self.hidden_size, use_bias=False, name='v', initializers=self.init)
            self._output_dense = snt.Linear(self.hidden_size, use_bias=False, name='output', initializers=self.init)

            self._sa = modules.SelfAttention()
            self._sa_laynorm = snt.LayerNorm()
            self._ff = snt.nets.MLP([self.hidden_size, self.hidden_size], activation=self.activ, initializers=self.init, activate_final=True)
            self._ff_laynorm = snt.LayerNorm()
            self._doub_dense = snt.Linear(2*self.hidden_size, use_bias=False, name='doub', initializers=self.init)
            # NOTE: I am writing the residual connections in different ways.  The FF was easy to use the snt.Residual, while this was difficult for the SA.
            # This can probably be cleaned
            # TODO: we may want to clean this up later so we can add multiple transformer blocks.  this just hard codes a single block right now

    def _build(self, inputs):
        graph = inputs['graph']
        nodes = self._input_dense(graph.nodes)

        q = self._q_dense(nodes)
        k = self._k_dense(nodes)
        v = self._v_dense(nodes)

        q = self.split_heads(q, self.FLAGS)
        k = self.split_heads(k, self.FLAGS)
        v = self.split_heads(v, self.FLAGS)

        attention_graph = self._sa(node_values=v, node_keys=k, node_queries=q, attention_graph=graph)

        attention_output = self.combine_heads(attention_graph.nodes, self.FLAGS)
        attention_output = self._output_dense(attention_output)

        sa_skip = nodes + attention_output  # residual/skip connection
        sa_normed = self._sa_laynorm(sa_skip)  # apply layer norm

        ff_skip = sa_normed + self._ff(sa_normed)  # residual/skip connection
        ff_normed = self._ff_laynorm(ff_skip)  # apply layer norm

        # nodes to global aggregator with graph
        if self.FLAGS['tf_gate']:
            ff_normed_doub = self._doub_dense(ff_normed)
            # TODO: try raw activation to make sure this is not just helping because of more weights
            weights = tf.tanh(ff_normed_doub[:,:self.hidden_size])
            vals = tf.sigmoid(ff_normed_doub[:,self.hidden_size:])
            gated = weights * vals
            out_graph = attention_graph.replace(nodes=gated)
        else:
            out_graph = attention_graph.replace(nodes=ff_normed)

        reducer = {'sum': tf.unsorted_segment_sum, 'max': blocks.unsorted_segment_max_or_zero}[self.FLAGS['tf_reducer']]
        agg = blocks.NodesToGlobalsAggregator(reducer=reducer)
        agged = agg(out_graph)
        agged = agged / tf.cast(out_graph.n_node, tf.float32)[:,None]
        return agged

class MLPEmbed(Module):
    """MLP network used"""
    def __init__(self, FLAGS, init=None, activ=None, name="MLPEmbed"):
        super().__init__(FLAGS=FLAGS, init=init, activ=activ, name=name)
        self.hidden_size = self.FLAGS['mlp_hidden_size']
        with self._enter_variable_scope():
            self._mlp = snt.nets.MLP([self.hidden_size]*self.FLAGS['num_mlp_layers'], activation=self.activ, initializers=self.init, activate_final=True)

    def _build(self, inputs):
        array = inputs['array']
        array = tf.reshape(array, [-1, 20])
        return self._mlp(array)

class ConvEmbed(Module):
    """CNN used in full method"""
    def __init__(self, FLAGS, init=None, activ=None, use_gformer=False, conv_flatten='flatten', is_training_ph=None, name='ConvEmbed'):
        super().__init__(FLAGS=FLAGS, init=init, activ=activ, name=name)
        self.use_gformer = use_gformer
        self.conv_flatten = conv_flatten
        if is_training_ph is not None:
            self.is_training = is_training_ph
        else:
            self.is_training = self.FLAGS['is_training']
        self.is_training = self.FLAGS['is_training']

        with self._enter_variable_scope():
            if self.FLAGS['deeper_net']:
                self._conv = snt.nets.ConvNet2D(
                    output_channels=[self.FLAGS['conv_filters']]*4,
                    kernel_shapes=[3,3,3,3], 
                    strides=[2,2,2,2], 
                    paddings=['SAME']*4,
                    use_batch_norm=self.FLAGS['use_bn'],
                    batch_norm_config={'update_ops_collection': None},
                    activation=self.activ
                    )
            else:
                self._conv = snt.nets.ConvNet2D(
                    output_channels=[self.FLAGS['conv_filters']]*3,
                    kernel_shapes=[5,3,3], 
                    strides=[3,2,2], 
                    paddings=['SAME']*3,
                    use_batch_norm=self.FLAGS['use_bn'],
                    batch_norm_config={'update_ops_collection': None},
                    activation=self.activ
                    )

            if self.use_gformer:
                self._gformer = GraphFormer(self.FLAGS, activ=self.activ)
            else:
                self._linear = snt.Linear(self.FLAGS['cnn_hidden_size'], initializers=self.init)
                self._flatten = snt.BatchFlatten()

    def _build(self, inputs):
        conv_out = self._conv(inputs['image'], is_training=self.is_training)
        if self.use_gformer:
            # Apply MHDPA to CNN activations
            if self.FLAGS['cnn_gn_coord']:
                # concatenate the image coordinates to retain spatial information.  find it works a bit better
                tf_conv_shape = tf.shape(conv_out)[:-1]
                X, Y = tf.meshgrid(tf.linspace(-1.0,1.0,tf_conv_shape[1]), tf.linspace(-1.0,1.0,tf_conv_shape[1]))
                X = tf.tile(X[None,:,:], [tf_conv_shape[0], 1, 1])[...,None]
                Y = tf.tile(Y[None,:,:], [tf_conv_shape[0], 1, 1])[...,None]
                conv_out = tf.concat([conv_out, X, Y], axis=-1)

            conv_shape = conv_out.shape
            conv_out = tf.reshape(conv_out, [-1, conv_shape[1].value*conv_shape[2].value, conv_shape[3].value])
            graph = convert_nodes_to_graph_tuple(conv_out, self.FLAGS, attr_size=conv_shape[3].value)
            conv_out = self._gformer({'graph': graph})
        else:
            # different output approaches
            if self.conv_flatten == 'flatten' or self.conv_flatten == 'linear':
                conv_out = self._flatten(conv_out)
                if self.conv_flatten == 'linear':
                    conv_out = self.activ(conv_out)
                    conv_out = self._linear(conv_out)
            elif 'ssam' in self.conv_flatten:
                conv_out = tf.contrib.layers.spatial_softmax(conv_out)
                if 'linear' in self.conv_flatten:
                    conv_out = self._linear(conv_out)

        return conv_out


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class MDN_Head(Module):
    def __init__(self, FLAGS, init=None, activ=None, ndim=None, k=None, name="MDN_Head", conv=False):
        super().__init__(FLAGS=FLAGS, init=init, activ=activ, name=name)

        if ndim is not None:
            self.ndim = ndim
        else:
            self.ndim = 2

        self.k = k or self.FLAGS['mdn_k']

        with self._enter_variable_scope():
            self._modules = {}
            self._modules['locs'] = snt.Linear(self.ndim*self.k, initializers=self.init)
            self._modules['scales'] = snt.Linear(self.ndim*self.k, initializers=self.init)
            self._modules['logits'] = snt.Linear(self.k, initializers=self.init)

    def _build(self, inputs):
        locs = self._modules['locs'](inputs)
        log_scales = self._modules['scales'](inputs)
        logits = self._modules['logits'](inputs)

        scales = tf.nn.softplus(log_scales + softplus_inverse(1.0))

        locs = tf.reshape(locs, [-1, self.k, self.ndim])
        scales = tf.reshape(scales, [-1, self.k, self.ndim])
        logits = tf.reshape(logits, [-1, self.k])
        # reshape so that the first dim is the mixture, because we are doing to unstack them
        # also swap the batch size and the ones that come from the steps of this run
        # (K x N x D)
        mix_first_locs = tf.transpose(locs, [1, 0, 2])
        mix_first_scales = tf.transpose(scales, [1, 0, 2])

        outs = {'locs': locs, 'scales': scales, 'logits': logits}
        outs['flattened'] = flatten_mdn(logits, locs, scales, self.FLAGS)

        cat = tfd.Categorical(logits=logits)
        components = []
        eval_components = []
        for loc, scale in zip(tf.unstack(mix_first_locs), tf.unstack(mix_first_scales)):
            normal = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
            components.append(normal)
            eval_normal = tfd.MultivariateNormalDiag(loc=loc[...,:2], scale_diag=scale[...,:2])
            eval_components.append(eval_normal)
        mixture = tfd.Mixture(cat=cat, components=components)
        eval_cat = tfd.Categorical(logits=logits)
        eval_mixture = tfd.Mixture(cat=eval_cat, components=eval_components)
        outs['mixture'] = mixture
        outs['eval_mixture'] = eval_mixture
        return outs

class EncoderConv(Module):
    """for a vae"""
    def __init__(self, FLAGS, init=None, activ=None, name='EncoderConv', is_training_ph=None):
        super().__init__(FLAGS=FLAGS, init=init, activ=activ, name=name)
        if is_training_ph is not None:
            self.is_training = is_training_ph
        else:
            self.is_training = self.FLAGS['is_training']
        self.is_training = self.FLAGS['is_training']

        with self._enter_variable_scope():
            if self.FLAGS['deeper_net']:
                self._conv = snt.nets.ConvNet2D(
                    output_channels=[self.FLAGS['conv_filters']]*4,
                    kernel_shapes=[3,3,3,3], 
                    strides=[2,2,2,2], 
                    paddings=['SAME']*4,
                    use_batch_norm=self.FLAGS['use_bn'],
                    batch_norm_config={'update_ops_collection': None},
                    activation=self.FLAGS['activ']
                    )
            else:
                self._conv = snt.nets.ConvNet2D(
                    output_channels=[16, 32, 32], 
                    kernel_shapes=[5, 5, 5], 
                    strides=[3, 3, 3], 
                    paddings=['SAME']*3, 
                    activation=self.activ,
                    initializers=self.init,
                    use_batch_norm=self.FLAGS['use_bn'],
                    batch_norm_config={'update_ops_collection': None},
                )
            self._flatten = snt.BatchFlatten()
            self._linear = snt.Linear(self.FLAGS['tf_hidden_size'], initializers=self.init)

    def _build(self, inputs, conv_flatten='flatten'):
        conv_out = self._conv(inputs, is_training=self.is_training)
        if conv_flatten == 'flatten' or conv_flatten == 'linear':
            conv_out = self._flatten(conv_out)

            if conv_flatten == 'linear':
                conv_out = self.activ(conv_out)
                conv_out = self._linear(conv_out)

        elif 'ssam' in conv_flatten:
                conv_out = tf.contrib.layers.spatial_softmax(conv_out)
                if 'linear' in conv_flatten:
                    conv_out = self._linear(conv_out)
        return conv_out

class DecoderConvT(Module):
    """for a vae"""
    def __init__(self, FLAGS, init=None, activ=None, name='DecoderConvT', is_training_ph=None):
        super().__init__(FLAGS=FLAGS, init=init, activ=activ, name=name)
        if is_training_ph is not None:
            self.is_training = is_training_ph
        else:
            self.is_training = self.FLAGS['is_training']
        self.is_training = self.FLAGS['is_training']

        with self._enter_variable_scope():
            self._reshape = snt.BatchReshape([1, 1, self.FLAGS['vae_z_size']])
            self._upconv = snt.nets.ConvNet2DTranspose(
                output_channels=[32, 16, 3], 
                output_shapes=[None, None, None], 
                kernel_shapes=[7, 4, 3], 
                strides=[1,4,3], 
                paddings=['VALID', 'SAME', 'SAME'],
                activation=self.activ,
                initializers=self.init,
                use_batch_norm=self.FLAGS['use_bn'],
                batch_norm_config={'update_ops_collection': None},
            )

    def _build(self, inputs, conv_flatten='flatten'):
        original_shape = tf.shape(inputs)
        logits = self._upconv(self._reshape(inputs), is_training=self.is_training)
        logits = tf.reshape(logits, shape=tf.concat([original_shape[:-1], self.FLAGS['image_shape']], axis=0))
        return logits

class MixturePrior(Module):
    def __init__(self, k, size, FLAGS, init=None, activ=None, name='MixturePrior'):
        super().__init__(FLAGS=FLAGS, init=init, activ=activ, name=name)
        if self.init is not None:
            init = self.init['w']
        else:
            init = None
        self.k = k
        self.size = size
        with self._enter_variable_scope():
            self._loc = tf.get_variable(name='loc', shape=[self.k, self.size], initializer=init)
            self._raw_scale_diag = tf.get_variable(name='raw_scale_diag', shape=[self.k, self.size], initializer=init)
            self._mixture_logits = tf.get_variable(name='mixture_logits', shape=[self.k])
            
    def _build(self, inputs=None):
        return tfd.MixtureSameFamily(
            components_distribution=tfd.MultivariateNormalDiag(
                loc=self._loc, 
                scale_diag=tf.nn.softplus(self._raw_scale_diag)), 
                mixture_distribution=tfd.Categorical(logits=self._mixture_logits), name="prior")


