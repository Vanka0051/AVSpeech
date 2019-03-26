# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 12:43:43 2018

@author: vanka0051
"""

import cv2  
import scipy.misc
#import numpy as np

videodir = './media/frame15/《TED演讲》值得尊敬！一位残疾人讲述自己调整心态的过程-国语流畅.mp4'
vc = cv2.VideoCapture(videodir) #读入视频文件  
c=1  



if vc.isOpened(): #判断是否正常打开  
    rval , frame = vc.read()  
else:  
    rval = False  
  
timeF = 1  #视频帧计数间隔频率  
  
while rval:   #循环读取视频帧  
    rval, frame = vc.read()  
    #人脸检测，有人脸则保存
    #############################
#    cascPath = "haarcascade_frontalface_default.xml"
#    faceCascade = cv2.CascadeClassifier(cascPath)
#    image = frame
#    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    faces = faceCascade.detectMultiScale(
#        gray,
#        scaleFactor=1.1,
#        minNeighbors=5,
#        minSize=(30, 30),
#    )
#    
#    faces = np.array(faces)
#    ######################################
#    if faces.any() :
#        scipy.misc.imsave('/home/vanka0051/project/python/AVSpeech/media/pic/1/'+str(c)+'.jpg', frame) #存储为图像  
#        
        
        
    scipy.misc.imsave('./media/pic/12/'+str(c)+'.jpg', frame)
    c = c + 1  
    cv2.waitKey(1)  
vc.release()







   
