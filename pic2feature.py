# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:51:57 2018

@author: vanka0051
"""




model_path='./align/20170511-185253.pb'


import tensorflow as tf
from tensorflow.python.platform import gfile
import os
from scipy import misc
import numpy as np
import facenet
import align.detect_face
import scipy.io as scio
import scipy.io.wavfile as wav
from sklearn import preprocessing
import matplotlib.pyplot as plt






os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#jpg2_path='/home/vanka0051/project/github/facenet/facenet-master/src/2.jpg'


def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    with gfile.FastGFile(model_exp,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, input_map=input_map, name='')
#summaryWriter = tf.summary.FileWriter('/home/vanka0051/application/github/facenet/model/log/', graph)
        #b = sess.graph.get_tensor_by_name("conv/b:0")
       # print(output_graph_def)
        
        
        
        
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
#    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
        sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

#        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
#        with tf.device("/gpu:3"):    
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    #tmp_image_paths=copy.copy(image_paths)
    img_list = []
    imagetest=image_paths
    #for image in tmp_image_paths:
    img = misc.imread(os.path.expanduser(imagetest), mode='RGB')
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if len(bounding_boxes) < 1:
        image_paths.remove(imagetest)
        print("can't detect face, remove ", imagetest)
    #    continue
    det = np.squeeze(bounding_boxes[0,0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    prewhitened = facenet.prewhiten(aligned)
    img_list.append(prewhitened)
    images = np.stack(img_list)
    return images
    
def get_facefeature( jpg_path, model_path ):
    images = load_and_align_data(jpg_path, 160, 44, 1.0)
    
    load_model(model_path)
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    #embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    facenet_avgpool = tf.get_default_graph().get_tensor_by_name("InceptionResnetV1/Logits/AvgPool_1a_8x8/AvgPool:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    
                # Run forward pass to calculate embeddings
    feed_dict = { images_placeholder: images, phase_train_placeholder:False }
    
    sess=tf.Session()
#    with tf.device("/gpu:3"):  
    avgpool = sess.run(facenet_avgpool, feed_dict=feed_dict)
    return avgpool 


if __name__ == '__main__':
    
    
    
    
 
    num = 0
    x = 0 * 60 +7
    y = x + 3
    tmpdata1 = np.zeros([45, 1792])
    for i in range(x*15 , y*15):
        jpg1_path='./media/pic/12/'+str(i)+'.jpg'
        avgpool=get_facefeature( jpg1_path, model_path)
        tmpdata1[i-(x * 15) , :] = avgpool
        num = num + 1
        if num%10 == 0:
            print ''+str(num)+' is done\n'

    filepath = './media/audio/《TED演讲》值得尊敬！一位残疾人讲述自己调整心态的过程-国语流畅.wav'
    fs, audio = wav.read(filepath)
    audio = preprocessing.scale(audio)
    audio1 = audio[16000*(x): 16000*(y)]
    del audio
    scio.savemat('./mat/12/12_'+str(x)+'_'+str(y)+'.mat',
             {'video': tmpdata1,
              'audio': audio1
              })












    num = 0
    x = 0 * 60 + 10
    y = x + 3
    tmpdata1 = np.zeros([45, 1792])
    for i in range(x*15 , y*15):
        jpg1_path='./media/pic/12/'+str(i)+'.jpg'
        avgpool=get_facefeature( jpg1_path, model_path)
        tmpdata1[i-(x * 15) , :] = avgpool
        num = num + 1
        if num%10 == 0:
            print ''+str(num)+' is done\n'

    filepath = './media/audio/《TED演讲》值得尊敬！一位残疾人讲述自己调整心态的过程->国语流畅.wav'
    fs, audio = wav.read(filepath)
    audio = preprocessing.scale(audio)
    audio1 = audio[16000*(x): 16000*(y)]
    del audio
    scio.savemat('./mat/12/12_'+str(x)+'_'+str(y)+'.mat',
             {'video': tmpdata1,
              'audio': audio1
              })


