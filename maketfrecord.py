import soundfile as sf
import tensorflow as tf
import numpy as np
import os
import random
import sys



os.environ['CUDA_VISIBLE_DEVICES'] = '2'
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)


i = 0
count = 0
audiolong = np.array([])
saveid = 0
path = './mat/TIMIT/train/'
LOG = open('./mat/TIMIT/testdata.txt',"w")
x = tf.placeholder(tf.float32, shape = [None])
y = tf.contrib.signal.stft(x, 400, 160, 512)
z = tf.contrib.signal.inverse_stft(y, 400, 160, 512)



tfrecords_filename = './mat/tfrecords/audioonly/small.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)


for home, dirs, files in os.walk(path):
    print('test')

files.sort()

try:
    for name1 in files:
        audio1, fs = sf.read(os.path.join(path, name1))
        len1 = len(audio1)
        for name2 in random.sample(files[i : len(files)], len(files[i : len(files)])//2):
            audio2, fs = sf.read(os.path.join(path, name2))
            audio2 = audio2[3 * 16000:]
            len2 = len(audio2)
            if len1<len2:
                length = len1
                audio22 = audio2[0:length]
                audio11 = audio1
            else :
                length = len2
                audio11 = audio1[0:length]
                audio22 = audio2
            audiomix = audio11 + audio22
            part_num = length // 48000
		
            for part in range(part_num):
                count = count + 1
                audiomix1 = audiomix[part * 48000 : (part +1 ) * 48000]
                label1 = audio11[part * 48000 : (part +1 ) * 48000]
                label2 = audio22[part * 48000 : (part +1 ) * 48000]
                audiomix1 = sess.run(z, feed_dict = {x : audiomix1}).tostring()
                label1 = sess.run(z, feed_dict = {x : label1}).tostring()
                label2 = sess.run(z, feed_dict = {x : label2}).tostring()
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                    'label1': tf.train.Feature(bytes_list = tf.train.BytesList(value=[label1])),     
                    'label2': tf.train.Feature(bytes_list = tf.train.BytesList(value=[label2])),
                    'aud_feature' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [audiomix1]))
                          }))
                writer.write(example.SerializeToString())
                LOG.write('label1:' + name1 + ',  label2:'+name2 + ',  part_num: %i ' %(part) )
                LOG.flush()
                if count%10000 == 0:
                    print '%i is done /n' %count
                if count == 50000 :
                    writer.close()
                    sys.exit()
        i = i + 1
    writer.close()
except KeyboardInterrupt:
	writer.close()





