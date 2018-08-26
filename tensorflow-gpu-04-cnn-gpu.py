#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
import os
import time
import numpy as np
import tflearn
import tensorflow as tf
from numba import cuda

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

from tensorflow.python.client import device_lib
def print_gpu_info():
    for device in device_lib.list_local_devices():
        print(device.name, 'memory_limit', str(round(device.memory_limit/1024/1024))+'M', 
            device.physical_device_desc)
    print('=======================')

print_gpu_info()


DATA_PATH = "/Volumes/Cloud/DataSet"

mnist = tflearn.datasets.mnist.read_data_sets(DATA_PATH+"/mnist", one_hot=True)

config = tf.ConfigProto()
config.log_device_placement = True
config.allow_soft_placement = True

config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3

# Building convolutional network
net = tflearn.input_data(shape=[None, 28, 28, 1], name='input')
net = tflearn.conv_2d(net, 32, 5, weights_init='variance_scaling', activation='relu', regularizer="L2")
net = tflearn.conv_2d(net, 64, 5, weights_init='variance_scaling', activation='relu', regularizer="L2")
net = tflearn.fully_connected(net, 10, activation='softmax')
net = tflearn.regression(net,
                         optimizer='adam',    
                         learning_rate=0.01,
                         loss='categorical_crossentropy',
                         name='target')

# Training
model = tflearn.DNN(net, tensorboard_verbose=3)

start_time = time.time()
model.fit(mnist.train.images.reshape([-1, 28, 28, 1]),
          mnist.train.labels.astype(np.int32),
          validation_set=(
              mnist.test.images.reshape([-1, 28, 28, 1]),
              mnist.test.labels.astype(np.int32)
          ),
          n_epoch=1,
          batch_size=128,
          shuffle=True,
          show_metric=True,
          run_id='cnn_mnist_tflearn')

duration = time.time() - start_time
print('Training Duration %.3f sec' % (duration))

cuda.select_device(0)
cuda.close()
