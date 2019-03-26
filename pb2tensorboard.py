# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 17:02:14 2018

@author: vanka0051
"""
import tensorflow as tf
import model
from tensorflow.python.framework import graph_util



#if __name__ == '__main__':
    
#    face_feature=facefeature.get_facefeature(jpg_path, model_path)    
#    LOG = open("/home/hyli/Data/InternData/log_full_sgd_lr0002.txt", "w")
    
tf.reset_default_graph()
aud_frame = 298
img_frame = 75
batch_size= 5
is_training = True
cost_valid_best = 1000000
n_epochs = 1000
aud_feature =tf.placeholder(tf.float32, shape=[None, None, 257,2])#(batch_size,aud_frame,257,2)
img_feature =tf.placeholder(tf.float32, shape=[None, None, 1,1792])#(batch_size,img_frame,1,1792)
#aud_feature =tf.placeholder(tf.float32, shape=[batch_size, aud_frame, 257,2])#(batch_size,aud_frame,257,2)
#img_feature =tf.placeholder(tf.float32, shape=[batch_size, img_frame, 1,1792])#(batch_size,img_frame,1,1792)
      
label = tf.placeholder(tf.float32, shape=[None, 257*2])
lrate_p = tf.placeholder(tf.float32)
logdir = '/home/vanka0051/project/python/AVSpeech/log/'    







aud_output = model.aud_conv_layer(inputx=aud_feature,   
              batch_size=batch_size, aud_frame=aud_frame, is_training=is_training)

vis_output = model.vis_conv_layer(inputx=img_feature,   
              batch_size=batch_size, img_frame=img_frame, is_training=is_training) 
              
              
vis_out = vis_output.forward_conv()#(batch_size, img_frame, 1, 256)
aud_out = aud_output.forward_conv()#(batch_size, aud_frame, 257, 8)
vis_out = tf.reshape (vis_out, shape = [batch_size, img_frame, 256])
aud_out = tf.reshape (aud_out, shape = [batch_size, aud_frame, 257*8])
vis_out = model.upsample_withbatch(vis_out , img_frame, aud_frame, batch_size)
fus_out = tf.concat([aud_out , vis_out], axis = 2)
bilstm_output = model.bilstm_layer(fus_out, batch_size, aud_frame, is_training=is_training)


####################################################################

bilstm_out = bilstm_output.forward_BiLSTM()     #输出为[batch_size, aud_frame, 400]
                                                    #如何以batch通过FC层，batch是什么

####################################################################
fc_output = model.fc_layer( inputx= bilstm_out, is_training=is_training )
fc_out = fc_output.forward_fc()
final_out1 = tf.reduce_sum((label - fc_out)**2, 0)/5
final_out2 = tf.reduce_sum(final_out1, 1)/514
cost_fine = tf.reduce_mean(final_out2)



#    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#    with tf.control_dependencies(update_ops):    
#        
#        train_fine = tf.train.AdamOptimizer(
#        learning_rate=lrate_p, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost_fine)
 
train_fine = tf.train.AdamOptimizer(
    learning_rate=lrate_p, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost_fine) 
saver = tf.train.Saver()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
writer = tf.summary.FileWriter("/home/vanka0051/project/python/AVSpeech/log/",sess.graph)
