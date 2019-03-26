
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 18:14:06 2018
@author: vanka0051
"""


import tensorflow as tf
import numpy as np
import time
from Googlemodel import *
from tensorflow.python.framework import graph_util
import scipy.io as scio
import os
import model1
from sklearn.utils import shuffle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-model_name',default = 'Jan15', type = str)
parser.add_argument('-batch_size',default = 32,type=int)
parser.add_argument('-epoch_num' ,default = 50 ,type=int)
parser.add_argument('-how_many_batch_to_show_cost' ,default = 3000 ,type=int)
parser.add_argument('-cost_in_time' ,default = False ,type=bool)
parser.add_argument('-lrate', default = 0.0001, type=float)
parser.add_argument('-GPU_ID', default = '2', type=str)
args = parser.parse_args()
CUDA_ID = args.GPU_ID

###############################    set parameter   ###################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_ID


def parser_function(serialized_example):
    features = tf.parse_single_example(serialized_example,
    features={
                    'aud_feature': tf.FixedLenFeature([], tf.string),
                    'img_feature': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.string)
                                                    })
    label = tf.decode_raw(features['label'], tf.float32)
    img_feature = tf.decode_raw(features['img_feature'], tf.float32)
    aud_feature = tf.decode_raw(features['aud_feature'], tf.float32)
    return aud_feature, img_feature, label



tf.reset_default_graph()


batch_size = args.batch_size
epoch_num = args.epoch_num
model_name = args.model_name
how_many_batch_to_show_cost = args.how_many_batch_to_show_cost
cost_in_time = args.cost_in_time
learning_rate = args.lrate



tfrecords_filename = './mat/tfrecords/AVSpeech/train.tfrecords'
dataset =  tf.data.TFRecordDataset(tfrecords_filename)
dataset = dataset.map(parser_function)
dataset = dataset.shuffle(123)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat(epoch_num)
iterator = dataset.make_one_shot_iterator()
aud, img, lab  = iterator.get_next()


tfrecords_filename_test = './mat/tfrecords/AVSpeech/test.tfrecords'
dataset =  tf.data.TFRecordDataset(tfrecords_filename_test)
dataset = dataset.map(parser_function)
dataset = dataset.shuffle(123)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat(epoch_num * 20)
iterator = dataset.make_one_shot_iterator()
aud_test, img_test, lab_test  = iterator.get_next()



logdir = os.path.join('./log', model_name)
isExists = os.path.exists(logdir)
if not isExists:
    os.makedirs(logdir)
LOG = open( logdir  +'/log_full_lr0002.txt',"w")




formats = '| epoch {0:>10d} | cost_train {1:>15.9f} |  cost_valid {2:>15.9f} | time {3:>12.3f} |'
formatd = '+{0:-^18}+{0:-^28}+{0:-^29}+{0:-^19}+'.format('-')

aud_frame = 298
img_frame = 72
################################ placeholder ###################

is_training = tf.placeholder(tf.bool , shape = [])
aud_featureinput = tf.placeholder(tf.float32, shape=[None, 47920], name = 'aud_featureinput')
#aud_feature =tf.placeholder(tf.float32, shape=[None, None, 257,2])#(batch_size,aud_frame,257,2)
img_feature =tf.placeholder(tf.float32, shape=[None, None, 1,1792], name = 'img_feature')#(batch_size,img_frame,1,1792)
label = tf.placeholder(tf.float32, shape=[None, None], name = 'label')
lrate_p = tf.placeholder(tf.float32, name = 'learning_rate')
dropout_rate = tf.placeholder(tf.float32, name = 'dropout_rate')



########################### AVconstruction #############################
aud_feature_stft = tf.contrib.signal.stft(aud_featureinput, 400, 160, 512)
real_feat = tf.expand_dims(tf.real(aud_feature_stft), 3)
imag_feat = tf.expand_dims(tf.imag(aud_feature_stft), 3)
aud_feature = tf.concat([real_feat,imag_feat], 3)

aud_out = inference(aud_feature, is_training)
aud_out = tf.reshape (aud_out, shape = [-1, aud_frame, 257*8])


vis_output = vis_conv_layer(inputx=img_feature,   is_training=is_training)
vis_out = vis_output.forward_conv()#(batch_size, img_frame, 1, 256)
#vis_out = tf.reshape (vis_out, shape = [-1, img_frame, 256])
#vis_out = model1.upsample_withbatch(vis_out , img_frame, aud_frame, batch_size)
vis_out = tf.image.resize_images(vis_out, [aud_frame, 1], method = 1)
vis_out = tf.reshape(vis_out, [-1 , aud_frame, 256])




fus_out = tf.concat([aud_out , vis_out], axis = 2)
bilstm_output = model1.bilstm_layer(fus_out, tf.shape(fus_out)[0],  is_training=is_training)

bilstm_out = bilstm_output.forward_BiLSTM()


#fc_output = model1.fc_layer( inputx= bilstm_out, is_training= True, is_trainingtf = is_training, dropout_rate = dropout_rate )
#fc_output = fc_layer( inputx= bilstm_out, is_training= True, is_trainingtf = is_training, dropout_rate = dropout_rate )

fc_output = model1.fc_layer( inputx= bilstm_out, is_training= True,  dropout_rate = dropout_rate )
fc_out_real, fc_out_imag = fc_output.forward_fc()
tf.add_to_collection("regularization_losses",tf.contrib.layers.l2_regularizer(0.00004)(tf.get_default_graph().get_tensor_by_name("FC_layer3/dense/kernel:0")))
tf.add_to_collection("regularization_losses",tf.contrib.layers.l2_regularizer(0.00004)(tf.get_default_graph().get_tensor_by_name("FC_layer3/dense_1/kernel:0")))
tf.add_to_collection("regularization_losses",tf.contrib.layers.l2_regularizer(0.00004)(tf.get_default_graph().get_tensor_by_name("FC_layer2/dense/kernel:0")))
tf.add_to_collection("regularization_losses",tf.contrib.layers.l2_regularizer(0.00004)(tf.get_default_graph().get_tensor_by_name("FC_layer1/dense/kernel:0")))



########################### Cost in time domain###############################
#fc_out_real = -1 * tf.log( (1 / fc_out[:, :, 0:257]) - 1)
#fc_out_imag = -1 * tf.log( (1 / fc_out[:, :, 257:514]) - 1)
fc_out_real = -1 * tf.log( (1 / fc_out_real) - 1)
fc_out_imag = -1 * tf.log( (1 / fc_out_imag) - 1)




realpart = fc_out_real * tf.abs(aud_feature_stft)
imagpart = fc_out_imag * tf.abs(aud_feature_stft)

stftout =tf.complex(realpart, imagpart)
wavout = tf.contrib.signal.inverse_stft(stftout, 400 , 160, 512, name = 'waveout')
stft_label = tf.contrib.signal.stft(label, 400, 160, 512)
reallabel = tf.real(stft_label)
imaglabel = tf.imag(stft_label)



cost_fine_intime  = tf.reduce_sum((label - wavout)**2, 1)
cost_fine_intime = tf.reduce_mean(cost_fine_intime, name = 'cost_fine_in_time')
cost_fine_intime_regular = cost_fine_intime + tf.add_n(tf.get_collection('regularization_losses'))


cost_fine =tf.reduce_mean(tf.reduce_sum (
                                tf.reduce_sum(
                                tf.pow(realpart- reallabel, 2)
                               +tf.pow(imagpart- imaglabel,2),
                               1),
                               1), name = 'cost_fine')/2.0

cost_fine_regular = cost_fine + tf.add_n(tf.get_collection('regularization_losses'))


#update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
update_ops = tf.get_collection(UPDATE_OPS_COLLECTION)
with tf.control_dependencies(update_ops):
#        
    train_fine = tf.train.AdamOptimizer(
        learning_rate=lrate_p, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost_fine)
    train_fine_regular = tf.train.AdamOptimizer(
        learning_rate=lrate_p, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost_fine_regular)
    train_fine_intime = tf.train.AdamOptimizer(
        learning_rate=lrate_p, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost_fine_intime)
    train_fine_intime_regular = tf.train.AdamOptimizer(
            learning_rate=lrate_p, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost_fine_intime_regular)


config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
sess = tf.Session(config = config)
#sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()






#print ' _______________________________________________________________________________'
#print ' |                                                                             |'
#print ' |                    EPOCH: 0, Validation cost: %.10f                         |' %(Cost_validation)
#print ' |                                                                             |'
#print ' |                         Start training......                                |'
#print ' |_____________________________________________________________________________|'
#print ' _______________________________________________________________________________'


print formatd
print '|{0: ^97}|'.format('')
print '|{0: ^97}|'.format('Start   training')
print '|{0: ^97}|'.format('')
print formatd
print formats.format(0, 0, 0, 0)


cost_train = 0
cost_train_best = 1000000
cost_test = 0
cost_test_best = 1000000

time_start = time.time()
for epoch in range(1000000):
    lrate = learning_rate
    audtest, imgtest, labtest = sess.run([aud, img, lab])
    imgtest = np.reshape(imgtest, [-1, 72,1, 1792])
    sess.run(train_fine_regular,feed_dict={aud_featureinput : audtest,
                                  img_feature : imgtest,
                                  label       : labtest,
                                      lrate_p : lrate,
                                   is_training: True,
                                  dropout_rate: 0.8})




    tmp_cost_train = sess.run(cost_fine,feed_dict={aud_featureinput : audtest,
                                  img_feature : imgtest,
                                  label       : labtest,
                                  lrate_p : lrate,
                                  is_training: False,
                                  dropout_rate: 1.0})
    cost_train = tmp_cost_train + cost_train

    audtest, imgtest, labtest = sess.run([aud_test, img_test, lab_test])
    imgtest = np.reshape(imgtest, [-1, 72,1, 1792])

    tmp_cost_test = sess.run(cost_fine,feed_dict={aud_featureinput : audtest,
                                 img_feature : imgtest,  
                                 label       : labtest,
                                 lrate_p : lrate,
                                 is_training: False,
                                 dropout_rate: 1.0})
    cost_test = tmp_cost_test + cost_test
    if (epoch+1)%how_many_batch_to_show_cost == 0:
        time_end = time.time()
        cost_train = cost_train / 257.0 / 298.0/ (how_many_batch_to_show_cost * 1.0 )
        cost_test  = cost_test  /  257.0 / 298.0/ (how_many_batch_to_show_cost * 1.0 )
        print formats.format(epoch + 1 , cost_train, cost_test, (time_end - time_start))
        LOG.write(formats.format(epoch + 1 , cost_train, cost_test , (time_end - time_start))+ '\n')
        LOG.flush()
        if cost_train <cost_train_best :
            cost_train_best = cost_train
            saver.save(sess, os.path.join( logdir , model_name) , global_step = epoch + 1)
        cost_train = 0
        cost_test  = 0
        time_start = time.time()

LOG.close()
sess.close()

