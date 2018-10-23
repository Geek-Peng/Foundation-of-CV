# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 21:24:54 2017

@author: God鹏
"""

import pickle
import numpy
import matplotlib.pyplot as plt
from PIL import Image

file1='F:\\数据集\\cifar-10-python\\cifar-10-batches-py\\data_batch_1'
file2='F:\\数据集\\cifar-10-python\\cifar-10-batches-py\\batches.meta'
file3='F:\\数据集\\cifar-10-python\\cifar-10-batches-py\\test_batch'

def unpickle(file):
    fo=open(file,'rb')
    dict = pickle.load(fo,encoding='iso-8859-1')
    fo.close()   
    return dict

dict_train_batch1=unpickle(file1)

a=numpy.array(dict_train_batch1.get('data'))
print(a) 
c=a[0]
b=numpy.reshape(a[0],(3,32,32))
print(b)
b0=Image.fromarray(b[0])
b1=Image.fromarray(b[1])
b2=Image.fromarray(b[2])
img=Image.merge('RGB',(b0,b1,b2))
plt.figure()
plt.imshow(img)

