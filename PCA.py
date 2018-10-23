# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 08:17:44 2017

@author: God鹏
"""
#from  numpy import *
import numpy
from PIL import Image
import pylab
import pickle
import matplotlib.pyplot as plt

file1='F:\\数据集\\cifar-10-python\\cifar-10-batches-py\\data_batch_1'

def pca(x):
    
    num_data,dim=numpy.shape(x) 

    
    mean_x = x.mean(axis=0)
    x=x-mean_x
    
    if dim>num_data:
        M=numpy.dot(x,x.T) 
        e,EV=numpy.linalg.eigh(M)
        tmp=(numpy.dot(x.T,EV)).T 
        print(tmp)
        S=numpy.sqrt(e)[::-1] #算出来的特征值是从小到大排列的，所以逆转
        V=tmp[::-1] #由于最后的特征向量是我们所需要的，所以将tmp逆转
        for i in range(V.shape[1]):
            V[:,i]/=S
    else:
        U,S,V=numpy.linalg.svd(x)  
        V=V[:num_data] 
        
    return  V,S,mean_x

def unpickle(file):
    fo=open(file,'rb')
    dict = pickle.load(fo,encoding='iso-8859-1')
    fo.close()   
    return dict

def transtoRGBimg(numpyarray):  
    b0=Image.fromarray(numpyarray[0])
    b1=Image.fromarray(numpyarray[1])
    b2=Image.fromarray(numpyarray[2])
    img=Image.merge('RGB',(b0,b1,b2))
    return img
def transtoGrayimg(numpyarray):
    R=numpyarray[0]
    G=numpyarray[1]
    B=numpyarray[2]
    img=R*0.3+G*0.59+B*0.11
    return img
    
data_dict=unpickle(file1)
imarray=numpy.array(data_dict.get('data'))
partimarry=imarray[0:20]

V,S,immean=pca(partimarry)
immean=numpy.uint8(numpy.array(immean)) 
plt.figure()
plt.gray()
pylab.subplot(2,4,1)
a=numpy.reshape(immean,(3,32,32))
img=transtoRGBimg(a)
plt.imshow(img)
for i in range(1,8):    
    pylab.subplot(2,4,i+1)
    imgv=transtoGrayimg(numpy.reshape(V[i],(3,32,32)))
    plt.imshow(imgv)

plt.show()





