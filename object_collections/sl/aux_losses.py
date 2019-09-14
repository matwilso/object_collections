import tensorflow as tf
from tensorflow_probability import distributions as tfd
from object_collections.sl.util import image_tile_summary

def mask_state(state):
    """
    Masking out extra state when there are less than the maximum number of objects present
    """

    state_list = state['array'][...,:2]
    state_shape = tf.shape(state_list)
    bs, num_nodes, attr_size = state_shape[0], state_shape[1], state_shape[2]
    tstate = tf.transpose(state_list, [1, 0, 2])

    masks = tf.cast(tf.not_equal(tstate, -1.0), tf.float32) 
    # TODO: fix this so it only triggers when there are multiple -1.0's.
    mask0 = masks[...,0]
    mask0_count = tf.reduce_sum(mask0, axis=0)
    # Check to make sure these line up.  
    # This is important to check for bugs which arise from how the data is packed and normalized
    mask_sum = tf.reduce_sum(masks, axis=-1)

    # TODO: fix this assertion.  It turns out that there are some cases where -1.0 is exactly is happening naturally
    #assert_op = tf.Assert(tf.reduce_all(tf.logical_or(tf.equal(mask_sum, tf.cast(attr_size, tf.float31)), tf.equal(mask_sum, 0))), [mask_sum])

    return mask0, mask0_count, tstate

def mdn_loss(state, mdn_vals, FLAGS):
    """
    Lstate in the paper 

    Maximum likelihood estimation of MDN and true state

    state['array'] is (batch x max_num_in_graph x attr_size)
    """
    mask0, mask0_count, tstate = mask_state(state)

    tshape = tf.shape(tstate)
    N, BS, D = tshape[0], tshape[1], tshape[2]

    mdn_logp = mdn_vals['mixture'].log_prob(tstate)
    nlogp = mask0 * -mdn_logp
    loss = tf.reduce_mean(nlogp, axis=0) * (tf.cast(N, tf.float32) / mask0_count)
    loss = tf.reduce_mean(loss)

    return loss

def mdn_metrics(state, mdn_vals, FLAGS):
    """
    ops to summarize and visualize MDN training 
    """
    
    # produce values for contour plots
    with tf.variable_scope('eval'):
        X, Y = tf.meshgrid(tf.linspace(-1.0,1.0,100), tf.linspace(-1.0,1.0,100))
        stacked = tf.stack([X,Y], axis=-1)[:,:,None,:]
        evalZ = mdn_vals['eval_mixture'].log_prob(stacked)

    eval_summaries = []
    with tf.name_scope('mdn'):
        #eval_summaries.append(tf.summary.scalar('loss', loss))
        eval_summaries.append(tf.summary.scalar('min_logits', tf.reduce_min(mdn_vals['logits'][0])))
        eval_summaries.append(tf.summary.scalar('max_logits', tf.reduce_max(mdn_vals['logits'][0])))
        eval_summaries.append(tf.summary.scalar('median_logits', tfd.percentile(mdn_vals['logits'], 50.0)))
    summary = tf.summary.merge(eval_summaries)
    #eval_vals = {'state': state, 'X': X, 'Y': Y, 'Z': evalZ, 'logits': mdn_vals['logits']}
    eval_vals = {'state': state, 'summary': summary, 'X': X, 'Y': Y, 'Z': evalZ, 'logits': mdn_vals['logits']}
    #eval_vals = {'state': state, 'summary': summary, 'loss': loss, 'X': X, 'Y': Y, 'Z': evalZ, 'logits': mdn_vals['logits']}
    #self.eval_vals = {'state': self.train_ds.state, 'samples': samples, 'summary': summary, 'loss': loss, 'X': X, 'Y': Y, 'Z': evalZ, 'logits': mdn_vals['logits']}
    return eval_vals

def vae_loss_and_metrics(state, vae_vals, FLAGS):
    """loss and summaries for VAE/autoencoder model"""

    # whether or not we use canonicalized images for training the AE 
    if FLAGS['use_canonical']:
        distortion = -vae_vals['phat'].log_prob(state['canonical'])
    else:
        distortion = -vae_vals['phat'].log_prob(state['image'])

    # approximate kl divergence trying to match approx_posterior to latent_prior 
    latent_prior = vae_vals['prior']
    rate = (vae_vals['q'].log_prob(vae_vals['q_sample']) - latent_prior.log_prob(vae_vals['q_sample']))

    if FLAGS['enforce_prior']:  
        elbo_local = -(FLAGS['vae_b']*rate + distortion)
    else:
        elbo_local = -distortion
    
    elbo = tf.reduce_mean(elbo_local)
    loss = -elbo

    eval_vals = {'rate': tf.reduce_mean(rate), 'distortion': tf.reduce_mean(distortion)}

    # define summaries
    eval_summaries = []
    with tf.name_scope('vae'):
        eval_summaries.append(image_tile_summary('input', tf.to_float(state['image']), rows=1, cols=8))
        recon_mean = vae_vals['phat'].mean()[:8]
        recon_clean = tf.round(recon_mean)
        eval_summaries.append(image_tile_summary('recon/mean', recon_mean, rows=1, cols=8))

    summary = tf.summary.merge(eval_summaries)
    eval_vals.update({'state': state, 'summary': summary})
    return loss, eval_vals

