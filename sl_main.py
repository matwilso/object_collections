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
from object_collections.sl.viz import plot
from tensorflow_probability import distributions as tfd

def main():
    t = Trainer(FLAGS)
    t.run()

if __name__ == "__main__":
    main()
