#!/usr/bin/env python

import tensorflow as tf

print('tensorflow', tf.__version__)

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

config = tf.ConfigProto()
config.log_device_placement = True  # 是否打印设备分配日志
config.allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
with tf.Session(config=config) as sess:
    # Runs the op.
    print(sess.run(c))
