import numpy as np
import scipy.io as scio
import os


#path="/media/data1/wangjc/AVSpeech/data/train/"



#aud = np.zeros(shape =[2484,298, 257, 2])
#img = np.zeros(shape =[2484, 45, 1, 1792])
#lab = np.zeros(shape =[2484, 298, 257])

aud = np.zeros(shape =[2484/2,298, 257, 2])
img = np.zeros(shape =[2484/2, 45, 1, 1792])
lab = np.zeros(shape =[2484/2, 298, 257 * 2])



x = 0
path1 ="/media/zhangjiandong/data1/wangjc/AVSpeech/mat/trainCNN/"

#path1 ="/media/data1/wangjc/AVSpeech/mat/trainIRM/"
#path2 = "/media/data1/wangjc/AVSpeech/mat/batch/alldata.mat"
#partest = scio.loadmat(path2)
for _, _ ,name in os.walk(path1):
    print 'test'
for name1 in name:
    par = scio.loadmat(path1 + name1)
    aud_feature = np.array(par['aud_feature'], dtype='float32')
    img_feature = np.array(par['img_feature'], dtype='float32')
    label = np.array(par['label'], dtype='float32')
    aud_feature = np.expand_dims(aud_feature, axis=0)
    img_feature = np.expand_dims(img_feature, axis=0)
    img_feature = np.expand_dims(img_feature, axis=2)
    label = np.expand_dims(label, axis=0)
    aud[x , :, :, :] = aud_feature
    img[x , :, :, :] = img_feature
#    lab[x , :, :] = label.transpose()
    lab[x , :, :] = label

    x = x +1

scio.savemat('/media/zhangjiandong/data1/wangjc/AVSpeech/mat/batchCNN/alldata.mat',
    {'label': lab,
     'aud_feature': aud,
     'img_feature': img                
     })

