import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

from config import Config

import datetime
import numpy as np
import os
import time

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]

tf.app.flags.DEFINE_integer('input_size', 224, "input image size")


activation = tf.nn.relu
bottleneck = False
x = tf.placeholder(tf.float32, shape = [4,298,257,2])
num_blocks = [2 ,2 ,2,2]
num_classes = None
is_training = tf.placeholder(tf.bool,shape = [])


use_bias = False


def inference(x, is_training,
              num_classes=None,
              num_blocks=[2, 2, 2, 2],  # defaults to 50-layer network
              use_bias=False, # defaults to using batch norm
              bottleneck=False):
    c = Config()
    c['bottleneck'] = bottleneck
#    c['is_training'] = tf.convert_to_tensor(is_training,
#                                            dtype='bool',
#                                            name='is_training')
    c['ksize'] = 3
    c['stride'] = 1
    c['use_bias'] = use_bias
    c['fc_units_out'] = num_classes
    c['num_blocks'] = num_blocks
    c['stack_stride'] = 2

    with tf.variable_scope('scale1'):
        c['conv_filters_out'] = 64
        c['ksize'] = 7
        c['stride'] = 2
        x = conv(x, c)
#        x = tf.contrib.layers.batch_norm(x, is_training= is_training)
        x = bn(x, is_training)
        x = activation(x)

    with tf.variable_scope('scale2'):
        x = _max_pool(x, ksize=3, stride=2)
        c['num_blocks'] = num_blocks[0]
        c['stack_stride'] = 1
        c['block_filters_internal'] = 64
        x = stack(x, c, is_training)

    with tf.variable_scope('scale3'):
        c['num_blocks'] = num_blocks[1]
        c['block_filters_internal'] = 128
        c['stack_stride'] = 2
        x = stack(x, c, is_training)

    with tf.variable_scope('scale4'):
        c['num_blocks'] = num_blocks[2]
        c['block_filters_internal'] = 256
        x = stack(x, c, is_training)
#
    with tf.variable_scope('scale5'):
        c['num_blocks'] = num_blocks[3]
        c['block_filters_internal'] = 512
        x = stack(x, c, is_training)
   
#    with tf.variable_scope('last'):
#        c['conv_filters_out'] = 8
#        c['ksize'] = 1
#        c['stride'] = 1
#        x = conv(x, c)
#        x = tf.contrib.layers.batch_norm(x, is_training= is_training)
#        x = bn(x, is_training)
#        x = activation(x)



    # post-net
#    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

#    if num_classes != None:
#        with tf.variable_scope('fc'):
#            x = fc(x, c)


    with tf.variable_scope('conv_transpose0'):
        x = conv_transpose(x, ksize = 3 , stride = 2, 
        filters_out = 512, output_len = 19, 
        output_wid = 17)
        x = bn(x, is_training)
        x = tf.nn.relu(x)

    with tf.variable_scope('conv_transpose1'):
        x = conv_transpose(x, ksize = 3 , stride = 2, 
        filters_out = 256, output_len = 38, 
        output_wid = 33)
        x = bn(x, is_training)
        x = tf.nn.relu(x)


    with tf.variable_scope('conv_transpose2'):
        x = conv_transpose(x, ksize = 3 , stride = 2, 
        filters_out = 128, output_len = 75, 
        output_wid = 65)
        x = bn(x, is_training)
        x = tf.nn.relu(x) 

    with tf.variable_scope('conv_transpose3'):
        x = conv_transpose(x, ksize = 3 , stride = 2, 
        filters_out = 64, output_len = 149, 
        output_wid = 129)
        x = bn(x, is_training)
        x = tf.nn.relu(x)    


    with tf.variable_scope('conv_transpose4'):
        x = conv_transpose(x, ksize = 7 , stride = 2, 
        filters_out = 8, output_len = 298, 
        output_wid = 257)
        x = bn(x, is_training)

        x = tf.nn.relu(x)


    return x





def stack(x, c, is_training):
    for n in range(c['num_blocks']):
        s = c['stack_stride'] if n == 0 else 1
        c['block_stride'] = s
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, c, is_training)
    return x





def block(x, c, is_training):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed. 
    # That is the case when bottleneck=False but when bottleneck is 
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    m = 4 if c['bottleneck'] else 1
    filters_out = m * c['block_filters_internal']

    shortcut = x  # branch 1

    c['conv_filters_out'] = c['block_filters_internal']

    if c['bottleneck']:
        with tf.variable_scope('a'):
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            x = conv(x, c)
#            x = tf.contrib.layers.batch_norm(x, is_training= is_training)
            x = bn(x, is_training)
            x = activation(x)

        with tf.variable_scope('b'):
            x = conv(x, c)
#            x = tf.contrib.layers.batch_norm(x, is_training= is_training )
            x = bn(x, is_training)

            x = activation(x)

        with tf.variable_scope('c'):
            c['conv_filters_out'] = filters_out
            c['ksize'] = 1
            assert c['stride'] == 1
            x = conv(x, c)
#            x = tf.contrib.layers.batch_norm(x, is_training = is_training)
            x = bn(x, is_training)

    else:
        with tf.variable_scope('A'):
            c['stride'] = c['block_stride']
            assert c['ksize'] == 3
            x = conv(x, c)
#            x = tf.contrib.layers.batch_norm(x, is_training = is_training )
            x = bn(x, is_training)
            x = activation(x)

        with tf.variable_scope('B'):
            c['conv_filters_out'] = filters_out
            assert c['ksize'] == 3
            assert c['stride'] == 1
            x = conv(x, c)
#            x = tf.contrib.layers.batch_norm(x, is_training= is_training )
            x = bn(x, is_training)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or c['block_stride'] != 1:
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            c['conv_filters_out'] = filters_out
            shortcut = conv(shortcut, c)
#            x = tf.contrib.layers.batch_norm(x, is_training= is_training )
            x = bn(x, is_training)


    return activation(x + shortcut)







def fc(x, fc_units_out):
    num_units_in = x.get_shape()[1]
    num_units_out = fc_units_out
    weights_initializer = tf.truncated_normal_initializer(
        stddev=FC_WEIGHT_STDDEV)

    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer)
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def conv(x, c):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)

    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')







def bn(x,  is_training):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

#    if c['use_bias']:
#        bias = _get_variable('bias', params_shape,
#                             initializer=tf.zeros_initializer)
#        return x + bias


    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer)
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer)

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer,
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #x.set_shape(inputs.get_shape()) ??

    return x







class vis_conv_layer(object):
    def __init__(self, inputx, is_training):

        self.is_training = is_training
        self.inputx =inputx

        self.shape1 = [7, 1, 1792 , 256]
#        self.weights1 = tf.get_variable("vis_weights1", shape=self.shape1)
        self.weights1 = _get_variable("vis_weights1", shape = self.shape1, initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV), weight_decay=0.00004)

        self.shape2 = [5, 1, 256 , 256]
#        self.weights2 = tf.get_variable("vis_weights2", shape=self.shape2)

        self.weights2 = _get_variable("vis_weights2", shape = self.shape2, initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV), weight_decay=0.00004)


        self.shape3 = [5, 1, 256 , 256]
#        self.weights3 = tf.get_variable("vis_weights3", shape=self.shape3)

        self.weights3 = _get_variable("vis_weights3", shape = self.shape3, initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV), weight_decay=0.00004)


        self.shape4 = [5, 1, 256 , 256]
#        self.weights4 = tf.get_variable("vis_weights4", shape=self.shape4)

        self.weights4 = _get_variable("vis_weights4", shape = self.shape4, initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV), weight_decay=0.00004)


        self.shape5 = [5, 1, 256 , 256]
#        self.weights5 = tf.get_variable("vis_weights5", shape=self.shape5)

        self.weights5 = _get_variable("vis_weights5", shape = self.shape5, initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV), weight_decay=0.00004)


        self.shape6 = [5, 1, 256 , 256]
#        self.weights6 = tf.get_variable("vis_weights6", shape=self.shape6)

        self.weights6 = _get_variable("vis_weights6", shape = self.shape6, initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV), weight_decay=0.00004)


    def forward_conv(self):


        with tf.variable_scope("vis_conv_layer1" ) as scope:

            conv1 = tf.nn.convolution(input=self.inputx, filter=self.weights1,
                                     dilation_rate=(1,1) , padding='SAME', name='conv')
#            BN1 = tf.contrib.layers.batch_norm(conv1, is_training= self.is_training )

            BN1 = bn(conv1 , self.is_training)
            layer1= tf.nn.relu(BN1, name= 'relu')

        with tf.variable_scope("vis_conv_layer2" ) as scope:
            conv2 = tf.nn.convolution(input=layer1, filter=self.weights2,
                                     dilation_rate=(1,1) , padding='SAME', name='conv')
#            BN2 = tf.contrib.layers.batch_norm(conv2, is_training= self.is_training )

            BN2 = bn(conv2, self.is_training)

            layer2= tf.nn.relu(BN2, name= 'relu')

        with tf.variable_scope("vis_conv_layer3" ) as scope:
            conv3 = tf.nn.convolution(input=layer2, filter=self.weights3,
                                     dilation_rate=(2,1) , padding='SAME', name='conv')
#            BN3 = tf.contrib.layers.batch_norm(conv3, is_training= self.is_training )
            BN3 = bn(conv3, self.is_training)
            layer3= tf.nn.relu(BN3, name= 'relu')

        with tf.variable_scope("vis_conv_layer4" ) as scope:
            conv4 = tf.nn.convolution(input=layer3, filter=self.weights4,
                                     dilation_rate=(4,1) , padding='SAME', name='conv')
#            BN4 = tf.contrib.layers.batch_norm(conv4, is_training= self.is_training )

            BN4 = bn(conv4, self.is_training)

            layer4= tf.nn.relu(BN4, name= 'relu')

        with tf.variable_scope("vis_conv_layer5" ) as scope:
            conv5 = tf.nn.convolution(input=layer4, filter=self.weights5,
                                     dilation_rate=(8,1) , padding='SAME', name='conv')
#            BN5 = tf.contrib.layers.batch_norm(conv5, is_training= self.is_training )

            BN5 = bn(conv5,  self.is_training)

            layer5= tf.nn.relu(BN5, name= 'relu')

        with tf.variable_scope("vis_conv_layer6" ) as scope:
            conv6 = tf.nn.convolution(input=layer5, filter=self.weights6,
                                     dilation_rate=(16,1) , padding='SAME', name='conv')
#            BN6 = tf.contrib.layers.batch_norm(conv6, is_training= self.is_training )

            BN6 = bn(conv6, self.is_training)

            self.layer6= tf.nn.relu(BN6, name= 'relu')


            return self.layer6




def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')






#def conv(x):
#    ksize = 7
#    stride = 2
#    filters_out = 8
#    filters_in = x.get_shape()[-1]
#    shape = [ksize, ksize, filters_in, filters_out]
#    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
#    weights = _get_variable('weights',
#                            shape=shape,
#                            dtype='float',
#                            initializer=initializer,
#                            weight_decay=CONV_WEIGHT_DECAY)
#    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')




def conv_transpose(x, ksize, stride, filters_out, output_len, output_wid):
    filters_in = x.get_shape()[-1]
    batch_size = tf.shape(x)[0]
    shape = [ksize, ksize, filters_out, filters_in]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return  tf.nn.conv2d_transpose(x , weights, output_shape = [batch_size,output_len , output_wid, filters_out], strides = [1 , stride, stride, 1], padding = 'SAME')







class fc_layer(object):
    def __init__(self, inputx, is_training, is_trainingtf,dropout_rate):
        self.inputx = inputx
        self.is_training = is_training
        self.dropout_rate = dropout_rate
        self.is_trainingtf = is_trainingtf
    def forward_fc(self):
        with tf.variable_scope("FC_layer1") as scope:
            fc_1 = tf.layers.dense(self.inputx, units = 600, activation = tf.nn.relu, trainable = self.is_training)
            fc_1 = bn(fc_1, self.is_trainingtf)
            fc_1 = tf.nn.dropout(fc_1, self.dropout_rate)
        with tf.variable_scope("FC_layer2") as scope:
            fc_2 = tf.layers.dense(fc_1, units = 600, activation = tf.nn.relu, trainable = self.is_training)
            fc_2 = bn(fc_2, self.is_trainingtf)
            fc_2 = tf.nn.dropout(fc_2, self.dropout_rate)
        with tf.variable_scope("FC_layer3") as scope:
            fc_3 = tf.layers.dense(fc_2, units = 257 * 2, activation = tf.nn.sigmoid, trainable = self.is_training)
#        with tf.variable_scope("FC_layer4") as scope:            
#            fc_4 = tf.layers.dense(fc_3, units = 257*2, activation = tf.nn.sigmoid, trainable = self.is_training)            
            return fc_3

