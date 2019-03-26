import tensorflow as tf
import numpy as np
import os
import scipy.io as scio

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
#config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

x = 0
y = 0
i = 0
num = 0

x = tf.placeholder(tf.float32, shape = [None])
y = tf.contrib.signal.stft(x, 400, 160, 512)
z = tf.contrib.signal.inverse_stft(y, 400, 160, 512)
#path = './mat/test/'
path = './mat/train/'
tfrecords_filename = './mat/tfrecords/audioonly/train1.tfrecords'
#tfrecords_filename = './mat/tfrecords/AVSpeech/test.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

for _,_,name in os.walk(path):
    print 'test'
try:
    for name1 in name:
        par1 = scio.loadmat(path + name1)
        audio1 = np.squeeze(np.array(par1['audio'], dtype='float32'))
#        video1 = np.array(par1['video'], dtype='float32')
        for name2 in name[i : len(name)]:
            if (name2[0:2] != name1[0:2]) :
                par2 = scio.loadmat(path + name2)
                audio2 = np.squeeze(np.array(par2['audio'], dtype='float32'))
#                video2 = np.array(par2['video'], dtype='float32')
                mix = audio1 + audio2
                label1 = sess.run(z, feed_dict = {x : audio1}).tostring()
                label2 = sess.run(z, feed_dict = {x : audio2}).tostring()
                audiomix = sess.run(z, feed_dict = {x : mix}).tostring()
#                example = tf.train.Example(features=tf.train.Features(
#                  feature={
#                    'label': tf.train.Feature(bytes_list = tf.train.BytesList(value=[label1])),
#                    'aud_feature' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [audiomix])),
#                    'img_feature' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [video1.tostring()]))
#                                            }))
#                writer.write(example.SerializeToString())
#                example = tf.train.Example(features=tf.train.Features(
#                              feature={
#                           'label': tf.train.Feature(bytes_list = tf.train.BytesList(value=[label2])),
#                           'aud_feature' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [audiomix])),
#                           'img_feature' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [video2.tostring()]))
#                                               }))
                example = tf.train.Example(features=tf.train.Features(
                          feature={
                          'label1': tf.train.Feature(bytes_list = tf.train.BytesList(value=[label1])),
                          'label2': tf.train.Feature(bytes_list = tf.train.BytesList(value=[label2])),
                          'aud_feature' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [audiomix]))
                          }))
                writer.write(example.SerializeToString())
                num = num +1 
                if num%10000 == 0:
                    print '%i is done /n' %num
        i = i +1 
    writer.close()
except KeyboardInterrupt:
    writer.close()



