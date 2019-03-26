import numpy as np
import scipy.io as scio
import os

aud  = np.zeros(shape = [800, 298, 257, 2])
img  = np.zeros(shape = [800, 45, 1, 1792])
lab  = np.zeros(shape = [800, 298, 257 * 2])
path = './mat/trainaudio/'
num  = 0
x    = 0

for _,_,name in os.walk(path):
    print 'test'

for name1 in name:
    par = scio.loadmat(path + name1)
    aud_feature = np.array(par['aud_feature'], dtype='float32')
    img_feature = np.array(par['img_feature'], dtype='float32')
    label = np.array(par['label'], dtype='float32')
    aud_feature = np.expand_dims(aud_feature, axis=0)
    img_feature = np.expand_dims(img_feature, axis=0)
    img_feature = np.expand_dims(img_feature, axis=2)
    label = np.expand_dims(label, axis=0)
    aud[num , :, :, :] = aud_feature
    img[num , :, :, :] = img_feature
    lab[num , :, :] = label
    num = num + 1
    if num%800 == 0 :
        scio.savemat('./mat/batchaudio/'+str(x)+'.mat', 
           {'label': lab,
            'aud_feature': aud,
            'img_feature': img                
     })
        x = x + 1
        num = 0





