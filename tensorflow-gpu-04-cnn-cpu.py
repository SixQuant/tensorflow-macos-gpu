#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
import os
import time
import numpy as np
import tflearn
import tensorflow as tf

#tf.logging.set_verbosity(tf.logging.ERROR)

DATA_PATH = "/Volumes/Cloud/DataSet"

mnist = tflearn.datasets.mnist.read_data_sets(DATA_PATH+"/mnist", one_hot=True)

config = tf.ConfigProto()
config.log_device_placement = True  # 是否打印设备分配日志
config.allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备

config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

with tf.device('/cpu'):
  # Building convolutional network
  net = tflearn.input_data(shape=[None, 28, 28, 1], name='input')  # 输入层 28*28 灰度图
  net = tflearn.conv_2d(net, 32, 5, weights_init='variance_scaling', activation='relu', regularizer="L2")  # 卷积层 输出 32 个特征，使用 5*5 卷积核
  net = tflearn.conv_2d(net, 64, 5, weights_init='variance_scaling', activation='relu', regularizer="L2")  # 卷积层 输出 64 个特征，使用 5*5 卷积核
  net = tflearn.fully_connected(net, 10, activation='softmax')     # 输出层 10 个分类，对应 softmax_cross_entropy_with_logits
  net = tflearn.regression(net,
                           optimizer='adam',                  # 对应 AdamOptimizer
                           learning_rate=0.01,
                           loss='categorical_crossentropy',   # 对应 softmax_cross_entropy_with_logits
                           name='target')

  model = tflearn.DNN(net, tensorboard_verbose=3)

# Training
start_time = time.time()
model.fit(mnist.train.images.reshape([-1, 28, 28, 1]), 
          mnist.train.labels.astype(np.int32), 
          validation_set=(
              mnist.test.images.reshape([-1, 28, 28, 1]), 
              mnist.test.labels.astype(np.int32)
          ),
          #validation_set=0.1, # 10% 数据用于验证
          n_epoch=1,              # 完整数据集投喂次数，太多或太少会导致过拟合或欠拟合
          batch_size=128,         # 每次训练获取的样本数
          shuffle=True,           # 是否对数据进行洗牌
          show_metric=True,       # 是否显示学习过程中的准确率
          run_id='cnn_mnist_tflearn')

duration = time.time() - start_time
print('Training Duration %.3f sec' % (duration))