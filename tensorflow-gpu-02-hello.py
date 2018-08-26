#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
from numba import cuda

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

config = tf.ConfigProto()
config.log_device_placement = True

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
with tf.Session(config=config) as sess:
    # Runs the op.
    print(sess.run(c))

cuda.select_device(0)
cuda.close()
