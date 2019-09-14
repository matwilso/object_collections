import io
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import tensorflow as tf

from object_collections.sl.constant import W, H

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
    return tf.summary.image(name, pack_images(tensor, rows, cols), max_outputs=1)

# Plotter functions
PLOT_FUNCS = {}
def register_plotter(func):
    PLOT_FUNCS[func.__name__] = func
    def func_wrapper(images, **kwargs):
        return func(images, **kwargs)
    return func_wrapper

def plot(mode, vals, FLAGS, itr=0, save=True, return_data=False, show=False):
    func = PLOT_FUNCS[mode]
    path = func(vals, FLAGS, itr=itr)

    data = None
    if save:
        plt.savefig(path)
    if return_data:
        fig = plt.get_current_fig_manager() 
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if show:
        plt.show()

    plt.close()
    return data

@register_plotter
def arr(arr, FLAGS, itr=None):
    plt.imshow(arr, cmap='binary')

@register_plotter
def in_out_vae(vals, FLAGS, itr=0):
    vae_title = '{}-vae.png'.format(itr)
    os.makedirs(FLAGS['plot_path'], exist_ok=True)
    vae_path = os.path.join(FLAGS['plot_path'], vae_title) 
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(vals['img1'])#, cmap='binary')
    ax2.imshow(vals['img2'])#, cmap='binary')
    return vae_path

@register_plotter
def contour(vals, FLAGS, itr=0):
    X, Y, Z, state = vals['X'], vals['Y'], vals['Z'], vals['state']
    prob_title = '{}-prob.png'.format(itr)
    os.makedirs(FLAGS['plot_path'], exist_ok=True)
    prob_path = os.path.join(FLAGS['plot_path'], prob_title) 
    #plt.axis('off')
    #plt.contour(X, Y, Z, alpha=0.5)
    ## mask out because some states don't exist.
    ## TODO: make this more uniform across stuff
    #mask0 = np.not_equal(state, -1.0)[:,0]
    #mask1 = np.not_equal(state, -1.0)[:,1]
    #assert np.all(mask0 == mask1)
    #plt.scatter(state[mask0[0]][...,0], state[mask0[0]][...,1], color='C1')
    #def tplot(temp, a1=1.0, a2=1.0, colors='white'):
    #    scaled = np.exp(Z / temp) / np.sum(np.exp(Z / temp))
    #    plt.tight_layout()
    #    plt.imshow(scaled)
    #    arr = ((state + 1) / 2) *  100 
    #    plt.contour(scaled, alpha=a1, colors=colors)
    #    plt.scatter(arr[...,0], arr[...,1], marker='x', color='C1', alpha=a2)
    #    plt.savefig('ttest.png')

    contour = plt.contourf(X, Y, Z, levels=np.arange(-28, 8, step=4))
    scat = plt.scatter(state[...,0], state[...,1], marker='x', color='C1')
    plt.colorbar(contour)


    #scaled = np.exp(Z / 10.0)
    #plt.imshow(scaled)
    #state = ((state + 1) / 2) *  100 
    #plt.scatter(state[...,0], state[...,1], marker='x', color='C1')
    #plt.contour(scaled, alpha=1.0, colors='white')

    #plt.title(prob_title)
    return prob_path

@register_plotter
def samples(vals, FLAGS, itr=0):
    samples = vals['samples']
    sample_title = '{}-sample.png'.format(itr)
    sample_path = os.path.join(FLAGS['plot_path'], sample_title) 

    sns.jointplot(samples[:,0,0], samples[:,0,1], kind='hex', color='#4cb391', xlim=(-1.0,1.0), ylim=(-1.0,1.0))
    return sample_path

@register_plotter
def shapes(vals, FLAGS, itr=None):
    dg = vals['dg']
    ax = plt.gca(aspect='equal', xlim=W, ylim=H)
    rect = mpatches.Rectangle((0,0), W, H, color='C0')
    ax.add_patch(rect)

    objs = dg.__next__()
    
    for o in objs['shapes']:
        o.plot(ax)

