import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import animation

import object_collections.rl.util as rl_util
from object_collections.define_flags import FLAGS
from object_collections.sl.data import Dataset
from object_collections.sl.viz import image_tile_summary, plot
from object_collections.rl.data import Dataset as RL_Dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def main():
    #self.train_ds = Dataset(self.FLAGS)
    action_dist = rl_util.convert_to_dict_dist(FLAGS['action_space'].spaces)
    obs_vectorizer = rl_util.convert_to_dict_dist(FLAGS['observation_space'].spaces)

    #FLAGS['bs'] = 64
    #FLAGS['filenames'] = list(reversed(FLAGS['filenames']))
    #FLAGS['filenames'] = FLAGS['filenames'][90:]

    train_ds = RL_Dataset(obs_vectorizer, FLAGS)

    sess = tf.InteractiveSession()
    sess.run(train_ds.iterator.initializer)

    def update_plot(i, state, scat):
        print(i)
        scat.set_offsets(state[i,:,:2])
        return scat,


    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    #anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    #plt.show()
    import ipdb; ipdb.set_trace()

    while True:
        state = sess.run(train_ds.state)['array']
        #import ipdb; ipdb.set_trace()
        #plt.scatter(state[:,0], state[:,1])

        numframes = FLAGS['bs']
        numpoints = 10
        fig = plt.figure()
        scat = plt.scatter(np.linspace(-1,1,numpoints), np.linspace(-1,1,numpoints))

        ani = animation.FuncAnimation(fig, update_plot, frames=range(numframes), fargs=(state, scat))
        plt.show()

if __name__ == '__main__':
    main()
