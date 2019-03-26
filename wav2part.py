# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 14:03:33 2018

@author: vanka0051
"""

import scipy.io.wavfile as wav
import scipy.io as scio
import stftme.stftme as stftpro

import matplotlib.pyplot as plt
from sklearn import preprocessing

filepath = '/home/vanka0051/project/python/AVSpeech/media/audio/wav/【TED演讲】比智商和情商更重要的品质——Grit。Grit可译为“坚毅”，但其-国语流畅.wav'

fs, audio = wav.read(filepath)

audio = audio[:, 0]
#audio0_45to0_48 = preprocessing.scale(audio[16000*(45-1)+1: 16000*(48-1)+1])
#audio1_34to1_37 = preprocessing.scale(audio[16000*(94-1)+1: 16000*(97-1)+1])
#audio2_13to2_16 = preprocessing.scale(audio[16000*(133-1)+1: 16000*(136-1)+1])
#audio2_16to2_19 = preprocessing.scale(audio[16000*(136-1)+1: 16000*(139-1)+1])
audio0_50to0_53 = (audio[16000*(50-1)+1: 16000*(53-1)+1])
audio2_18to2_21 = (audio[16000*(138-1)+1: 16000*(141-1)+1])
audio2_23to2_26 = (audio[16000*(143-1)+1: 16000*(146-1)+1])
audio2_26to2_29 = (audio[16000*(146-1)+1: 16000*(149-1)+1])
del audio



scio.savemat('/home/vanka0051/project/python/AVSpeech/media/fusion/2/audio.mat',
             {'audio0_50to0_53':audio0_50to0_53,
              'audio2_18to2_21':audio2_18to2_21,
              'audio2_23to2_26':audio2_23to2_26,
              'audio2_26to2_29':audio2_26to2_29})














