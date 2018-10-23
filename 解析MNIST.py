# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:20:27 2018

@author: God鹏
"""
import struct
import numpy  
trainimgdir='F:\\数据集\\MNIST\\train-images-idx3-ubyte\\train-images.idx3-ubyte'
trainlabdir='F:\\数据集\\MNIST\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte'


def decode_idx3_ubyte(file):
    bin_file=open(file,'rb')
    bin_buf=bin_file.read()
    
    #文件头解析
    index=0
    magic,numimg,rows,cols=struct.unpack_from('>iiii',bin_buf,index)
    index+=struct.calcsize('>iiii')    
    imgsize=rows*cols
    fmt_img='>'+str(imgsize)+'B'
    #图片解析
    imgs=numpy.empty((numimg,rows,cols))    
    for i in range(numimg):
        im=struct.unpack_from(fmt_img,bin_buf,index)
        index+=struct.calcsize(fmt_img)
        imgs[i]=numpy.array(im).reshape((rows,cols))
    
    return imgs

def decode_idx1_ubyte(file):
    bin_file=open(file,'rb')
    bin_buf=bin_file.read()
    
    #文件头解析
    index=0
    magic,numlab=struct.unpack_from('>ii',bin_buf,index)  #前面要返回几个参数就写几个i
    index+=struct.calcsize('>ii')
    fmt_lab='>B'
    labs=numpy.empty((numlab))
    for i in range(numlab):
        labs[i]=struct.unpack_from(fmt_lab,bin_buf,index)[0]
        index +=struct.calcsize(fmt_lab)
    return labs
    
trainlabarray=decode_idx1_ubyte(trainlabdir)
trainimgarray=decode_idx3_ubyte(trainimgdir)