# AVSpeech
AVSpeech compiled in tensorflow

Ariel E , Inbar M , Oran L , et al. Looking to listen at the cocktail party[J]. ACM Transactions on Graphics, 2018, 37(4):1-11.

different from this paper, we use resnet-18 to replace the dilated CNNs to save memory for GPU and enlarge the batch size. However, to compensate the differences between audio sampling and video sampling. Deconvolution layers are used. As a result, the output size is the same rather than variable like the model in the paper.  

resnet model comes from https://github.com/ry/tensorflow-resnet

facenet is used to pre-process the audio part.  
facenet model comes from https://github.com/davidsandberg/facenet


there are two main codes  
Googletrain_audioonly.py is used to train a model without the video part.  
Googletrain.py is used to train a model with both video and audio parts.
