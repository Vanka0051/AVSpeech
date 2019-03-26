
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
import model1audioonly
from sklearn.utils import shuffle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-model_name' )
parser.add_argument('-batch_size',default = 50,type=int)
parser.add_argument('-epoch_num' ,default = 50 ,type=int)
parser.add_argument('-how_many_batch_to_show_cost' ,default = 3578 ,type=int)
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
            'label1': tf.FixedLenFeature([], tf.string),
			'label2': tf.FixedLenFeature([], tf.string)
        })
    label1 = tf.decode_raw(features['label1'], tf.float32)
    label2 = tf.decode_raw(features['label2'], tf.float32)
    aud_feature = tf.decode_raw(features['aud_feature'], tf.float32)
    return aud_feature, label1, label2



tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config = config)


batch_size = args.batch_size
epoch_num = args.epoch_num
model_name = args.model_name
how_many_batch_to_show_cost = args.how_many_batch_to_show_cost
cost_in_time = args.cost_in_time
learning_rate = args.lrate



tfrecords_filename = './mat/tfrecords/audioonly/train.tfrecords'
dataset =  tf.data.TFRecordDataset(tfrecords_filename)
dataset = dataset.map(parser_function)
dataset = dataset.shuffle(123)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat(epoch_num)
iterator = dataset.make_one_shot_iterator()
aud, lab1, lab2  = iterator.get_next()


tfrecords_filename_test = './mat/tfrecords/audioonly/test.tfrecords'
dataset =  tf.data.TFRecordDataset(tfrecords_filename_test)
dataset = dataset.map(parser_function)
dataset = dataset.shuffle(123)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat(epoch_num * 6)
iterator = dataset.make_one_shot_iterator()
aud_test, lab1_test, lab2_test  = iterator.get_next()




logdir = os.path.join('./log', model_name)
isExists = os.path.exists(logdir)
if not isExists:
    os.makedirs(logdir) 


LOG = open( logdir  +'/log_full_lr0002.txt',"w")


formats = '| batch {0:>10d} | cost_train {1:>15.9f} |  cost_valid {2:>15.9f} | time {3:>12.3f} |'
formatd = '+{0:-^18}+{0:-^28}+{0:-^29}+{0:-^19}+'.format('-')
aud_frame = 298

################################ placeholder ###################

is_training = tf.placeholder(tf.bool , shape = [], name = 'is_training')
aud_featureinput = tf.placeholder(tf.float32, shape=[None, 47920], name = 'aud_featureinput')#(batch_size,aud_frame,257,2)
label1 = tf.placeholder(tf.float32, shape = [None, 47920], name = 'label1')
label2 = tf.placeholder(tf.float32, shape = [None, 47920], name = 'label2')
lrate_p = tf.placeholder(tf.float32, name = 'learning_rate')
dropout_rate = tf.placeholder(tf.float32, name = 'dropout_rate')
moment = tf.placeholder(tf.float32, name = 'momentum')


########################### AVconstruction #############################

#stfttransform_x = tf.placeholder(tf.float32, shape = [None,47920])
#stfttransform_y = tf.contrib.signal.stft(stfttransform_x, 400, 160, 512)
#stfttransform_z = tf.contrib.signal.inverse_stft(stfttransform_y, 400, 160, 512)


aud_feature_stft = tf.contrib.signal.stft(aud_featureinput, 400, 160, 512)
real_feat = tf.expand_dims(tf.real(aud_feature_stft), 3)
imag_feat = tf.expand_dims(tf.imag(aud_feature_stft), 3)
aud_feature = tf.concat([real_feat,imag_feat], 3) 
aud_out = inference(aud_feature, is_training)
aud_out = tf.reshape (aud_out, shape = [-1, aud_frame, 257*8])


bilstm_output = model1audioonly.bilstm_layer(aud_out, tf.shape(aud_out)[0],  is_training=is_training)

bilstm_out = bilstm_output.forward_BiLSTM()


#fc_output = model1.fc_layer( inputx= bilstm_out, is_training= True, is_trainingtf = is_training, dropout_rate = dropout_rate )
#fc_output = fc_layer( inputx= bilstm_out, is_training= True, is_trainingtf = is_training, dropout_rate = dropout_rate )

fc_output = model1audioonly.fc_layer( inputx= bilstm_out, is_training= True,  dropout_rate = dropout_rate )
#fc_out = fc_output.forward_fc()
fc_out_real1, fc_out_imag1, fc_out_real2, fc_out_imag2 = fc_output.forward_fc()

tf.add_to_collection("regularization_losses",tf.contrib.layers.l2_regularizer(0.00004)(tf.get_default_graph().get_tensor_by_name("FC_layer3/dense/kernel:0")))
tf.add_to_collection("regularization_losses",tf.contrib.layers.l2_regularizer(0.00004)(tf.get_default_graph().get_tensor_by_name("FC_layer3/dense_1/kernel:0")))
tf.add_to_collection("regularization_losses",tf.contrib.layers.l2_regularizer(0.00004)(tf.get_default_graph().get_tensor_by_name("FC_layer3/dense_2/kernel:0")))
tf.add_to_collection("regularization_losses",tf.contrib.layers.l2_regularizer(0.00004)(tf.get_default_graph().get_tensor_by_name("FC_layer3/dense_3/kernel:0")))


tf.add_to_collection("regularization_losses",tf.contrib.layers.l2_regularizer(0.00004)(tf.get_default_graph().get_tensor_by_name("FC_layer2/dense/kernel:0")))
tf.add_to_collection("regularization_losses",tf.contrib.layers.l2_regularizer(0.00004)(tf.get_default_graph().get_tensor_by_name("FC_layer1/dense/kernel:0")))



########################### Cost in time domain###############################
#fc_out_real1 = -1 * tf.log( (1 / fc_out[:, :, 0:257]) - 1)
#fc_out_imag1 = -1 * tf.log( (1 / fc_out[:, :, 257:514]) - 1)
#fc_out_real2 = -1 * tf.log( (1 / fc_out[:, :, 514:514+257]) - 1)
#fc_out_imag2 = -1 * tf.log( (1 / fc_out[:, :, 514+257:514*2]) - 1)


fc_out_real1 = -1 * tf.log( (1 / fc_out_real1 - 1))
fc_out_imag1 = -1 * tf.log( (1 / fc_out_imag1 - 1))
fc_out_real2 = -1 * tf.log( (1 / fc_out_real2 - 1))
fc_out_imag2 = -1 * tf.log( (1 / fc_out_imag2 - 1))


realpart1 = fc_out_real1 * tf.abs(aud_feature_stft)
imagpart1 = fc_out_imag1 * tf.abs(aud_feature_stft)
realpart2 = fc_out_real2 * tf.abs(aud_feature_stft)
imagpart2 = fc_out_imag2 * tf.abs(aud_feature_stft)

if cost_in_time==False :
    stftout1 = tf.complex(realpart1, imagpart1)
    stftout2 = tf.complex(realpart2, imagpart2)
    wavout1 = tf.contrib.signal.inverse_stft(stftout1, 400, 160, 512, name ='wavout1')
    wavout2 = tf.contrib.signal.inverse_stft(stftout2, 400, 160, 512, name ='wavout2')
    stft_label1 = tf.contrib.signal.stft(label1, 400, 160, 512)
    stft_label2 = tf.contrib.signal.stft(label2, 400, 160, 512)
    reallabel1 = tf.real(stft_label1)
    imaglabel1 = tf.imag(stft_label1)
    reallabel2 = tf.real(stft_label2)
    imaglabel2 = tf.imag(stft_label2)
    cost1 = tf.reduce_sum(  tf.reduce_sum(tf.pow(realpart1-reallabel1,2),1)
                           +tf.reduce_sum(tf.pow(imagpart1-imaglabel1,2),1)
                           +tf.reduce_sum(tf.pow(realpart2-reallabel2,2),1)
                           +tf.reduce_sum(tf.pow(imagpart2-imaglabel2,2),1),1, name = 'cost1') 
    cost2 = tf.reduce_sum(  tf.reduce_sum(tf.pow(realpart2-reallabel1,2),1)
                           +tf.reduce_sum(tf.pow(imagpart2-imaglabel1,2),1)
                           +tf.reduce_sum(tf.pow(realpart1-reallabel2,2),1)
                           +tf.reduce_sum(tf.pow(imagpart1-imaglabel2,2),1),1, name = 'cost2') 
    idx = tf.cast(cost1>cost2,tf.float32)
    cost_fine = tf.reduce_mean(idx*cost2+(1-idx)*cost1, name = 'cost_fine')
    cost_fine_regular = tf.reduce_sum(cost_fine + tf.add_n(tf.get_collection('regularization_losses')), name = 'cost_fine_regular')
    

    cost1_in_time = tf.reduce_mean( tf.reduce_sum(tf.pow(wavout1-label1,2),1)
                                   +tf.reduce_sum(tf.pow(wavout2-label2,2),1)
                                                                  , name = 'cost1_in_time')
    cost2_in_time = tf.reduce_mean( tf.reduce_sum(tf.pow(wavout1-label2,2),1)
                                   +tf.reduce_sum(tf.pow(wavout2-label1,2),1)
                                                                  , name = 'cost2_in_time')
    idx_in_time = tf.cast(cost1_in_time > cost2_in_time,tf.float32)
    cost_fine_in_time = tf.reduce_sum(idx_in_time*cost2_in_time+(1-idx_in_time)*cost1_in_time, name = 'cost_fine_in_time')
    cost_fine_regular_in_time = cost_fine_in_time + tf.add_n(tf.get_collection('regularization_losses'))
    tf.summary.scalar('cost_fine', cost_fine)
    tf.summary.scalar('cost_fine_in_time', cost_fine_in_time)
    merge_summaries = tf.summary.merge_all(key = 'summaries')




if cost_in_time==True :
    stftout1 = tf.complex(realpart1, imagpart1)
    stftout2 = tf.complex(realpart2, imagpart2)
    wavout1 = tf.contrib.signal.inverse_stft(stftout1, 400 , 160, 512)
    wavout2 = tf.contrib.signal.inverse_stft(stftout2, 400, 160, 512)


    cost1 = tf.reduce_mean( tf.reduce_sum(tf.pow(wavout1-label1,2),1)
                               +tf.reduce_sum(tf.pow(wavout2-label2,2),1)
                               ,1) 
    cost2 = tf.reduce_mean( tf.reduce_sum(tf.pow(wavout1-label2,2),1)
                               +tf.reduce_sum(tf.pow(wavout2-label1,2),1)
                               ,1)   


    idx = tf.cast(cost1>cost2,tf.float32)
    cost_fine = tf.reduce_sum(idx*cost2+(1-idx)*cost1)
    cost_fine_regular = cost_fine + tf.add_n(tf.get_collection('regularization_losses'))


#update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
update_ops = tf.get_collection(UPDATE_OPS_COLLECTION)
with tf.control_dependencies(update_ops):
#        
    train_fine = tf.train.AdamOptimizer(
    learning_rate=lrate_p, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost_fine)
    train_fine_regular = tf.train.AdamOptimizer(
    learning_rate=lrate_p, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost_fine_regular)

    train_fine_in_time = tf.train.AdamOptimizer(
    learning_rate=lrate_p, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost_fine_in_time)
    train_fine_regular_in_time = tf.train.AdamOptimizer(
    learning_rate=lrate_p, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost_fine_regular_in_time)

    train_fine_moment = tf.train.MomentumOptimizer(lrate_p , moment  ).minimize(cost_fine_regular)



init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
train_writer = tf.summary.FileWriter(logdir , sess.graph)


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
    audtest, lab1test, lab2test = sess.run([aud, lab1, lab2])
    sess.run(train_fine_moment,feed_dict={
                                     moment : 0.8,
                                   aud_featureinput : audtest,
                                        label1      : lab1test,
                                        label2      : lab2test,
                                        lrate_p : lrate,
                                        is_training:True,
                                        dropout_rate: 0.8})
    tmp_cost_train = sess.run(cost_fine,feed_dict={aud_featureinput : audtest,
        label1      : lab1test,
        label2      : lab2test,
        is_training: False,
        dropout_rate: 1.0})
    cost_train = tmp_cost_train + cost_train

    audtest, lab1test, lab2test = sess.run([aud_test, lab1_test, lab2_test])
    tmp_cost_test = sess.run(cost_fine,feed_dict={aud_featureinput : audtest,
			label1      : lab1test,
			label2      : lab2test,
			is_training: False,
			dropout_rate: 1.0})
    cost_test = tmp_cost_test + cost_test
    
    if (epoch+1)%how_many_batch_to_show_cost == 0:
        time_end = time.time()
        cost_train = cost_train / 257.0 / 298.0/ (how_many_batch_to_show_cost * 1.0 )
        cost_test  = cost_test  / 257.0 / 298.0/ (how_many_batch_to_show_cost * 1.0 )
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













#test = sess.run(cost_fine,feed_dict={aud_featureinput : audtest,label1      : lab1test,label2      : lab2test,is_training: False,dropout_rate: 1.0})

