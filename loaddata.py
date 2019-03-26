# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 19:35:37 2018

@author: vanka0051
"""
import os
import scipy.io as scio
import numpy as np
from stftme.stftme import do_stft
from sklearn import preprocessing
import tensorflow as tf
#import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


x = 0
y = 0
i = 0
num = 0
x1 = tf.placeholder(tf.float32, shape = [None])
y1 = tf.contrib.signal.stft(x1, 400, 160, 512)
z1 = tf.contrib.signal.inverse_stft(y1, 400 ,160 ,512)


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.Session(config = config)


aud  = np.zeros(shape = [400, 298, 257, 2])
img  = np.zeros(shape = [400, 72, 1, 1792])
lab  = np.zeros(shape = [400, 47920])

path='./mat/test/'


feed = np.zeros((298, 257, 2))
LOG = open('./mat/testwav说明.txt', "w")

for _, _ ,name in os.walk(path):
    print 'test'

#for name2 in name1:
for name1 in name:
    par1 = scio.loadmat(path + name1)
    audio1 = np.squeeze(np.array(par1['audio'], dtype='float32').transpose())
    video1 = np.array(par1['video'], dtype='float32')

    for name2 in name[i : len(name)]:
        if (name2[0:2] != name1[0:2]) :
            par2 = scio.loadmat(path + name2)
            audio2 = np.squeeze(np.array(par2['audio'], dtype='float32').transpose())
            video2 = np.array(par2['video'], dtype='float32')
            
            mix = audio1 + audio2 
            mixstft = sess.run(y1 , feed_dict = {x1 : mix})
            audiolabel1 = sess.run(z1, feed_dict = {x1 : audio1})
            audiolabel2 = sess.run(z1, feed_dict = {x1 : audio2})
            mixreal = mixstft.real
            miximag = mixstft.imag
            feed[:, :, 0] = mixreal
            feed[:, :, 1] = miximag



            aud_feature = np.expand_dims(feed, axis=0)
            img_feature1 = np.expand_dims(video1, axis=0)
            img_feature1 = np.expand_dims(img_feature1, axis=2)
            img_feature2 = np.expand_dims(video2, axis=0)
            img_feature2 = np.expand_dims(img_feature2, axis=2)
 
            aud[num , :, :, :] = aud_feature
            img[num , :, :, :] = img_feature1
            lab[num , :] = audiolabel1
            num = num + 1 
 
            aud[num , :, :, :] = aud_feature
            img[num , :, :, :] = img_feature2
            lab[num , :] = audiolabel2
            num = num + 1 



            if num%400 == 0 :
                scio.savemat('./mat/testbatch/'+str(y)+'.mat',
                   {'label': lab,
                    'aud_feature': aud,
                    'img_feature': img
                      }) 
                y = y + 1
                num = 0

#            scio.savemat('./mat/testmat/'+str(x)+'_speaker1.mat',
	
#                {'label': audiolabel1,
#                 'aud_feature': feed,
#                 'img_feature': video1
#                 })



#            scio.savemat('./mat/testmat/'+str(x)+'_speaker2.mat',
#                {'label': audiolabel2,
#                 'aud_feature': feed,
#                 'img_feature': video2
#                 })

            LOG.write('对于'+str(x)+'来说speaker是：'+name1+'noise是：'+name2+'\n')
            LOG.flush()
            LOG.write('对于'+str(x+1)+'来说speaker是：'+name2+'noise是：'+name1+'\n')
            LOG.flush()

            x = x + 2
    i = i+1

            
            
            
            
            
