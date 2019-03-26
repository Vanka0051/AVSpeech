# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 22:10:03 2018

@author: vanka0051
"""
import tensorflow as tf



class aud_conv_layer(object):
    def __init__(self, inputx, is_training):

        self.inputx = inputx
        self.is_training= is_training
        
        self.shape1 = [1, 7, 2 , 96]
        self.weights1 = tf.Variable(tf.truncated_normal(self.shape1, stddev=0.05))
        
        self.shape2 = [7, 1, 96 , 96]
        self.weights2 = tf.Variable(tf.truncated_normal(self.shape2, stddev=0.05))
        
        self.shape3 = [5, 5, 96 , 96]
        self.weights3 = tf.Variable(tf.truncated_normal(self.shape3, stddev=0.05))
        
        self.shape4 = [5, 5, 96 , 96]
        self.weights4 = tf.Variable(tf.truncated_normal(self.shape4, stddev=0.05))
        
        self.shape5 = [5, 5, 96 , 96]
        self.weights5 = tf.Variable(tf.truncated_normal(self.shape5, stddev=0.05))
        
        self.shape6 = [5, 5, 96 , 96]
        self.weights6 = tf.Variable(tf.truncated_normal(self.shape6, stddev=0.05))
        
        self.shape7 = [5, 5, 96 , 96]
        self.weights7 = tf.Variable(tf.truncated_normal(self.shape7, stddev=0.05))
        
        self.shape8 = [5, 5, 96 , 96]
        self.weights8 = tf.Variable(tf.truncated_normal(self.shape8, stddev=0.05))
        
        self.shape9 = [5, 5, 96 , 96]
        self.weights9 = tf.Variable(tf.truncated_normal(self.shape9, stddev=0.05))
        
        self.shape10 = [5, 5, 96 , 96]
        self.weights10 = tf.Variable(tf.truncated_normal(self.shape10, stddev=0.05))
        
        self.shape11 = [5, 5, 96 , 96]
        self.weights11 = tf.Variable(tf.truncated_normal(self.shape11, stddev=0.05))
        
        self.shape12 = [5, 5, 96 , 96]
        self.weights12 = tf.Variable(tf.truncated_normal(self.shape12, stddev=0.05))
        
        self.shape13 = [5, 5, 96 , 96]
        self.weights13 = tf.Variable(tf.truncated_normal(self.shape13, stddev=0.05))
        
        self.shape14 = [5, 5, 96 , 96]
        self.weights14 = tf.Variable(tf.truncated_normal(self.shape14, stddev=0.05))
        
        self.shape15 = [1, 1, 96 , 8]
        self.weights15 = tf.Variable(tf.truncated_normal(self.shape15, stddev=0.05))
        
        
    def forward_conv(self):
        with tf.variable_scope("aud_conv_layer1") as scope:
            conv1 = tf.nn.convolution(input=self.inputx, filter=self.weights1, 
                                     dilation_rate=(1,1) , padding='SAME', name='conv')
            BN1 = tf.contrib.layers.batch_norm(conv1, is_training= self.is_training )
            layer1= tf.nn.relu(BN1, name= 'relu')

        with tf.variable_scope("aud_conv_layer2") as scope:
            conv2 = tf.nn.convolution(input=layer1, filter=self.weights2, 
                                     dilation_rate=(1,1) , padding='SAME', name='conv')
            BN2 = tf.contrib.layers.batch_norm(conv2, is_training= self.is_training )
            layer2= tf.nn.relu(BN2, name= 'relu')
                     
        with tf.variable_scope("aud_conv_layer3") as scope:
            conv3 = tf.nn.convolution(input=layer2, filter=self.weights3, 
                                     dilation_rate=(1,1) , padding='SAME', name='conv')
            BN3 = tf.contrib.layers.batch_norm(conv3, is_training= self.is_training )
            layer3= tf.nn.relu(BN3, name= 'relu')
            
        with tf.variable_scope("aud_conv_layer4") as scope:
            conv4 = tf.nn.convolution(input=layer3, filter=self.weights4, 
                                     dilation_rate=(2,1) , padding='SAME', name='conv')
            BN4 = tf.contrib.layers.batch_norm(conv4, is_training= self.is_training )
#            output4 = BN4 + layer2            
#            layer4= tf.nn.relu(output4, name= 'relu') 

            layer4 = tf.nn.relu(BN4, name = 'relu')           
            
        with tf.variable_scope("aud_conv_layer5") as scope:
            conv5 = tf.nn.convolution(input=layer4, filter=self.weights5, 
                                     dilation_rate=(4,1) , padding='SAME', name='conv')
            BN5 = tf.contrib.layers.batch_norm(conv5, is_training= self.is_training )
            layer5= tf.nn.relu(BN5, name= 'relu')

        with tf.variable_scope("aud_conv_layer6") as scope:
            conv6 = tf.nn.convolution(input=layer5, filter=self.weights6, 
                                     dilation_rate=(8,1) , padding='SAME', name='conv')
            BN6 = tf.contrib.layers.batch_norm(conv6, is_training= self.is_training )
#            output6 = BN6 + layer4            
#            layer6= tf.nn.relu(output6, name= 'relu') 

            layer6 = tf.nn.relu(BN6, name = 'relu')   

        with tf.variable_scope("aud_conv_layer7") as scope:
            conv7 = tf.nn.convolution(input=layer6, filter=self.weights7, 
                                     dilation_rate=(16,1) , padding='SAME', name='conv')
            BN7 = tf.contrib.layers.batch_norm(conv7, is_training= self.is_training )
            layer7= tf.nn.relu(BN7, name= 'relu')

        with tf.variable_scope("aud_conv_layer8") as scope:
            conv8 = tf.nn.convolution(input=layer7, filter=self.weights8, 
                                     dilation_rate=(32,1) , padding='SAME', name='conv')
            BN8 = tf.contrib.layers.batch_norm(conv8, is_training= self.is_training )
#            output8 = BN8 + layer6            
#            layer8= tf.nn.relu(output8, name= 'relu') 

            layer8 = tf.nn.relu(BN8, name = 'relu')   

        with tf.variable_scope("aud_conv_layer9") as scope:
            conv9 = tf.nn.convolution(input=layer8, filter=self.weights9, 
                                     dilation_rate=(1,1) , padding='SAME', name='conv')
            BN9 = tf.contrib.layers.batch_norm(conv9, is_training= self.is_training )
            layer9= tf.nn.relu(BN9, name= 'relu')

        with tf.variable_scope("aud_conv_layer10") as scope:
            conv10 = tf.nn.convolution(input=layer9, filter=self.weights10, 
                                     dilation_rate=(2,2) , padding='SAME', name='conv')
            BN10 = tf.contrib.layers.batch_norm(conv10, is_training= self.is_training )
#            output10 = BN10 + layer8            
#            layer10= tf.nn.relu(output10, name= 'relu')  

            layer10 = tf.nn.relu(BN10, name = 'relu')  

        with tf.variable_scope("aud_conv_layer11") as scope:
            conv11 = tf.nn.convolution(input=layer10, filter=self.weights11, 
                                     dilation_rate=(4,4) , padding='SAME', name='conv')
            BN11 = tf.contrib.layers.batch_norm(conv11, is_training= self.is_training )
            layer11= tf.nn.relu(BN11, name= 'relu')

        with tf.variable_scope("aud_conv_layer12") as scope:
            conv12 = tf.nn.convolution(input=layer11, filter=self.weights12, 
                                     dilation_rate=(8,8) , padding='SAME', name='conv')
            BN12 = tf.contrib.layers.batch_norm(conv12, is_training= self.is_training )
#            output12 = BN12 + layer10            
#            layer12= tf.nn.relu(output12, name= 'relu')   

            layer12 = tf.nn.relu(BN12, name = 'relu') 

        with tf.variable_scope("aud_conv_layer13") as scope:
            conv13 = tf.nn.convolution(input=layer12, filter=self.weights13, 
                                     dilation_rate=(16,16) , padding='SAME', name='conv')
            BN13 = tf.contrib.layers.batch_norm(conv13, is_training= self.is_training )
            layer13= tf.nn.relu(BN13, name= 'relu')

        with tf.variable_scope("aud_conv_layer14") as scope:
            conv14 = tf.nn.convolution(input=layer13, filter=self.weights14, 
                                     dilation_rate=(32,32) , padding='SAME', name='conv')
            BN14 = tf.contrib.layers.batch_norm(conv14, is_training= self.is_training )
#            output14 = BN14 + layer12            
#            layer14= tf.nn.relu(output14, name= 'relu')   

            layer14 = tf.nn.relu(BN14, name = 'relu') 

        with tf.variable_scope("aud_conv_layer15") as scope:
            conv15 = tf.nn.convolution(input=layer14, filter=self.weights15, 
                                     dilation_rate=(1,1) , padding='SAME', name='conv')
            BN15 = tf.contrib.layers.batch_norm(conv15, is_training= self.is_training)
            self.layer15= tf.nn.relu(BN15, name= 'relu')
    
            return self.layer15



#########################################################################


class vis_conv_layer_model1(object):
    def __init__(self, inputx, is_training):

        self.is_training = is_training
        self.inputx =inputx

        self.shape1 = [7, 1, 1792 , 256]
        self.weights1 = tf.get_variable("vis_weights1", shape=self.shape1)

        self.shape2 = [5, 1, 256 , 256]
        self.weights2 = tf.get_variable("vis_weights2", shape=self.shape2)
        
        self.shape3 = [5, 1, 256 , 256]
        self.weights3 = tf.get_variable("vis_weights3", shape=self.shape3)
        
        self.shape4 = [5, 1, 256 , 256]
        self.weights4 = tf.get_variable("vis_weights4", shape=self.shape4)
        
        self.shape5 = [5, 1, 256 , 256]
        self.weights5 = tf.get_variable("vis_weights5", shape=self.shape5)

        self.shape6 = [5, 1, 256 , 256]
        self.weights6 = tf.get_variable("vis_weights6", shape=self.shape6)

        
    def forward_conv(self):

            
        with tf.variable_scope("vis_conv_layer1" ) as scope:

            conv1 = tf.nn.convolution(input=self.inputx, filter=self.weights1, 
                                     dilation_rate=(1,1) , padding='SAME', name='conv')
            BN1 = tf.contrib.layers.batch_norm(conv1, is_training= self.is_training )

#            BN1 = bn(conv1, self.is_training)
            layer1= tf.nn.relu(BN1, name= 'relu')
            
        with tf.variable_scope("vis_conv_layer2" ) as scope:
            conv2 = tf.nn.convolution(input=layer1, filter=self.weights2, 
                                     dilation_rate=(1,1) , padding='SAME', name='conv')
            BN2 = tf.contrib.layers.batch_norm(conv2, is_training= self.is_training )

#            BN2 = bn(conv2, self.is_training)

            layer2= tf.nn.relu(BN2, name= 'relu')
                     
        with tf.variable_scope("vis_conv_layer3" ) as scope:
            conv3 = tf.nn.convolution(input=layer2, filter=self.weights3, 
                                     dilation_rate=(2,1) , padding='SAME', name='conv')
            BN3 = tf.contrib.layers.batch_norm(conv3, is_training= self.is_training )
#            BN3 = bn(conv3, self.is_training)

            layer3= tf.nn.relu(BN3, name= 'relu')
            
        with tf.variable_scope("vis_conv_layer4" ) as scope:
            conv4 = tf.nn.convolution(input=layer3, filter=self.weights4, 
                                     dilation_rate=(4,1) , padding='SAME', name='conv')
            BN4 = tf.contrib.layers.batch_norm(conv4, is_training= self.is_training )

#            BN4 = bn(conv4, self.is_training)

            layer4= tf.nn.relu(BN4, name= 'relu')            
            
        with tf.variable_scope("vis_conv_layer5" ) as scope:
            conv5 = tf.nn.convolution(input=layer4, filter=self.weights5, 
                                     dilation_rate=(8,1) , padding='SAME', name='conv')
            BN5 = tf.contrib.layers.batch_norm(conv5, is_training= self.is_training )

#            BN5 = bn(conv5, self.is_training)

            layer5= tf.nn.relu(BN5, name= 'relu')

        with tf.variable_scope("vis_conv_layer6" ) as scope:
            conv6 = tf.nn.convolution(input=layer5, filter=self.weights6, 
                                     dilation_rate=(16,1) , padding='SAME', name='conv')
            BN6 = tf.contrib.layers.batch_norm(conv6, is_training= self.is_training )

#            BN6 = bn(conv6, self.is_training)

            self.layer6= tf.nn.relu(BN6, name= 'relu')


            return self.layer6

####################################################################

class bilstm_layer(object):
    def __init__(self, inputx,  batch_size, is_training):
        
        self.batch_size=batch_size
 
        self.inputx = inputx
        self.is_training = is_training
    def forward_BiLSTM(self):
        with tf.variable_scope("BiLSTM_layer" ) as scope:

            
            cell_fw = tf.contrib.rnn.LSTMCell(num_units=200)
            cell_bw = tf.contrib.rnn.LSTMCell(num_units=200)
            init_statef = cell_fw.zero_state(self.batch_size, dtype=tf.float32) # 初始化全零 state
            init_stateb = cell_fw.zero_state(self.batch_size, dtype=tf.float32) # 初始化全零 state
            # tensor of shape: [max_time, batch_size, input_size]
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                        cell_bw, inputs= self.inputx, 
                        initial_state_fw = init_statef,
                        initial_state_bw = init_stateb)
            output_fw, output_bw = outputs
            # forward states, backward states
  #          output_state_fw, output_state_bw = output_states
 #           output_fb1 = tf.concat([output_fw, output_bw], 2)
 #           shape = output_fb1.get_shape().as_list()
#            output_fb = tf.reshape(output_fb1, [shape[0], shape[1], 2, int(shape[2] / 2)])
#            hidden = tf.reduce_sum(output_fb, 2)/2.0
#            hidden = output_fw
            hidden =  tf.concat([output_fw, output_bw], 2)
            output = tf.nn.relu(hidden)
                    
            return output

######################################################

class fc_layer(object):
    def __init__(self, inputx, is_training, dropout_rate):
        self.inputx = inputx
        self.is_training = is_training
        self.dropout_rate = dropout_rate 
#        self.is_trainingtf = is_trainingtf
    def forward_fc(self):
        with tf.variable_scope("FC_layer1") as scope:
            fc_1 = tf.layers.dense(self.inputx, units = 600, activation = tf.nn.relu, trainable = self.is_training , kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))
#            fc_1 = bn(fc_1, self.is_trainingtf)
            fc_1 = tf.nn.dropout(fc_1, self.dropout_rate)
        with tf.variable_scope("FC_layer2") as scope:            
            fc_2 = tf.layers.dense(fc_1, units = 600, activation = tf.nn.relu, trainable = self.is_training, kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))
#            fc_2 = bn(fc_2, self.is_trainingtf)
            fc_2 = tf.nn.dropout(fc_2, self.dropout_rate)
#        with tf.variable_scope("FC_layer3") as scope:            
#            fc_3 = tf.layers.dense(fc_2, units = 257 * 2 * 2, activation = tf.nn.sigmoid, trainable = self.is_training)
        with tf.variable_scope("FC_layer3") as scope:  
            fc_out_real1 = tf.layers.dense(fc_2, units = 257 , activation = tf.nn.sigmoid, trainable = self.is_training, kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))
            fc_out_imag1 = tf.layers.dense(fc_2, units = 257 , activation = tf.nn.sigmoid, trainable = self.is_training, kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))
            fc_out_real2 = tf.layers.dense(fc_2, units = 257 , activation = tf.nn.sigmoid, trainable = self.is_training, kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))
            fc_out_imag2 = tf.layers.dense(fc_2, units = 257 , activation = tf.nn.sigmoid, trainable = self.is_training, kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))            
            return fc_out_real1, fc_out_imag1, fc_out_real2, fc_out_imag2
########################################################################

#class fc_layer_test(object):
#    def __init__(self, inputx):
#        self.inputx = inputx
#        self.weights1 = tf.Variable(rng.uniform(low = -0.1,
#                        high = 0.1, size=(400, 600)).astype('float32'))
#        self.weights2 = tf.Variable(rng.uniform(low = -0.1,
#                        high = 0.1, size=(600, 600)).astype('float32'))
#        self.weights3 = tf.Variable(rng.uniform(low = -0.1,
#                        high = 0.1, size=(600, 600)).astype('float32'))                        
#        self.weights4 = tf.Variable(rng.uniform(low = -0.1,
#                        high = 0.1, size=(600, 257*2)).astype('float32'))                        
#        self.b1 = tf.Variable(np.zeros([600]).astype('float32'))                
#        self.b2 = tf.Variable(np.zeros([600]).astype('float32'))                
#        self.b3 = tf.Variable(np.zeros([600]).astype('float32'))                
#        self.b4 = tf.Variable(np.zeros([257*2]).astype('float32'))
#                
#    def forward_fc(self):
#        with tf.variable_scope("FC_layer1") as scope:
#            fc1 = tf.matmul(self.inputx, self.weights1) + self.b1
#            fc_1 = tf.nn.relu(fc1)
#        with tf.variable_scope("FC_layer2") as scope:            
#            fc2 = tf.matmul(fc_1, self.weights2) + self.b2
#            fc_2 = tf.nn.relu(fc2)
#        with tf.variable_scope("FC_layer3") as scope:            
#            fc3 = tf.matmul(fc_2, self.weights3) + self.b3
#            fc_3 = tf.nn.relu(fc3)       
#        with tf.variable_scope("FC_layer4") as scope:            
#            fc4 = tf.matmul(fc_3, self.weights4) + self.b4
#            fc_4 = tf.nn.relu(fc4)     
#            return fc_4        
        

def upsample(img, img_frame, aud_frame):  
    n = aud_frame/ img_frame + 1
    m = n -1
    mod1 = aud_frame - img_frame * m
    mod2 = img_frame - mod1
    tmp_list = []
    for i in range(mod1):
        for j in range(n):
            tmp_list.append(img[i,:])
    for i in range(mod2):
        for j in range(m):
            tmp_list.append(img[mod1 + i,:])
    output = tf.stack(tmp_list)
    return output
    
    
    
def upsample_withbatch(inputdata , img_frame, aud_frame , batch_size): 
    tmp_list = []
    for i in range(batch_size):
        tmp_data = inputdata [i , : , : ]
        tmp_data = upsample(tmp_data , img_frame, aud_frame)
        tmp_list.append(tmp_data)
    output = tf.stack(tmp_list)
    return output




def bn(x, is_training):
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
