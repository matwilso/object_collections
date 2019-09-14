#!/usr/bin/env python3

import itertools
import os
import random
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf

from object_collections.define_flags import FLAGS
from object_collections.sl.trainer import Trainer
import object_collections.rl.util as rl_util
from object_collections.rl.data import rollout_to_tf_record
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2

#FLAGS['bs'] = 1
FLAGS['use_image'] = 1
FLAGS['shuffle'] = 0
FLAGS['shuffle_files'] = 0

def create_tf_record(images):
    action_dist = rl_util.convert_to_dict_dist(FLAGS['action_space'].spaces)
    obs_vectorizer = rl_util.convert_to_dict_dist(FLAGS['observation_space'].spaces)

    idxs = obs_vectorizer.idxs()
    names = list(obs_vectorizer.dout_shape.keys())

    img_beg, img_end = idxs[names.index('image')], idxs[names.index('image')+1]
    sases = {'s': [], 'a': [], 's_next': []}

    for img in images:
        state = np.zeros(obs_vectorizer.out_shape)
        state[img_beg:img_end] = img.ravel()
        sases['s'].append(state)
        sases['a'].append(np.zeros(action_dist.out_shape))
        sases['s_next'].append(np.zeros(obs_vectorizer.out_shape))

    for key in sases: 
        sases[key] = np.stack(sases[key])[None]

    filename = rollout_to_tf_record(sases, obs_vectorizer, FLAGS, FLAGS['eval_image_path'])
    return filename

def load_images():
    path = FLAGS['eval_image_path']
    files = os.listdir(path)
    files = [f for f in files if 'jpg' in f]
    files = list(sorted(map(lambda x: os.path.join(path, x), files)))
    images = [plt.imread(f) for f in files]
    images = [cv2.resize(image, FLAGS['image_shape'][:2], interpolation=cv2.INTER_AREA) / 255.0 for image in images]
    #for img in images:
    #    plt.imshow(img); plt.show()
    return images

def main():
    images = load_images()
    filename = create_tf_record(images)
    print('Created tf record')

    t = Trainer(FLAGS)
    t.evaluate(filename)

if __name__ == "__main__":
    main()
