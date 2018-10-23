# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 08:57:39 2017

@author: God鹏

对cifar10进行分类,思路：
    分别根据lable求出每一个类的词汇，然后KNN
"""
import numpy
import matplotlib.pyplot  as plt
import cv2
import pickle
import Vocabulary   #应用前面我写的Vocabler.py

file1='F:\\数据集\\cifar-10-python\\cifar-10-batches-py\\data_batch_1'

def unpickle(file):
    f=open(file,'rb')
    dict=pickle.load(f,encoding='iso-8859-1')
    f.close()
    return dict

imgdict=unpickle(file1)
imglist0=[]
imglist1=[]
imglist2=[]
imglist3=[]
imglist4=[]
imglist5=[]
imglist6=[]
imglist7=[]
imglist8=[]
imglist9=[]
imgdata=imgdict.get('data')
imglabels=imgdict.get('labels')
for i in range(len(imglabels)):
    if(imglabels[i]==0):
        imglist0.append(imgdata[i])
    elif(imglabels[i]==1):
        imglist1.append(imgdata[i])
    elif(imglabels[i]==2):
        imglist2.append(imgdata[i])
    elif(imglabels[i]==3):
        imglist3.append(imgdata[i])
    elif(imglabels[i]==4):
        imglist4.append(imgdata[i])
    elif(imglabels[i]==5):
        imglist5.append(imgdata[i])
    elif(imglabels[i]==6):
        imglist6.append(imgdata[i])
    elif(imglabels[i]==7):
        imglist7.append(imgdata[i])
    elif(imglabels[i]==8):
        imglist8.append(imgdata[i])
    elif(imglabels[i]==9):
        imglist9.append(imgdata[i])
imgarr0=numpy.array(imglist0)
imgarr1=numpy.array(imglist1)
imgarr2=numpy.array(imglist2)
imgarr3=numpy.array(imglist3)
imgarr4=numpy.array(imglist4)
imgarr5=numpy.array(imglist5)
imgarr6=numpy.array(imglist6)
imgarr7=numpy.array(imglist7)
imgarr8=numpy.array(imglist8)
imgarr9=numpy.array(imglist9)

vocabulary0=Vocabulary.Vocabulary('voc0')
vocabulary1=Vocabulary.Vocabulary('voc1')
vocabulary2=Vocabulary.Vocabulary('voc2')
vocabulary3=Vocabulary.Vocabulary('voc3')
vocabulary4=Vocabulary.Vocabulary('voc4')
vocabulary5=Vocabulary.Vocabulary('voc5')
vocabulary6=Vocabulary.Vocabulary('voc6')
vocabulary7=Vocabulary.Vocabulary('voc7')
vocabulary8=Vocabulary.Vocabulary('voc8')
vocabulary9=Vocabulary.Vocabulary('voc9')
descriptionset0=[]
for i in range(len(imgarr0)):
    b=numpy.reshape(imgarr0[i],(3,32,32))
    cvimg=cv2.merge([b[0],b[1],b[2]])
    graimg=cv2.cvtColor(cvimg,cv2.COLOR_BGR2GRAY)
    descriptionset0.append(vocabulary0.getSIFT(graimg))
descriptionset1=[]
for i in range(len(imgarr1)):
    b=numpy.reshape(imgarr1[i],(3,32,32))
    cvimg=cv2.merge([b[0],b[1],b[2]])
    graimg=cv2.cvtColor(cvimg,cv2.COLOR_BGR2GRAY)
    descriptionset1.append(vocabulary1.getSIFT(graimg))
descriptionset2=[]
for i in range(len(imgarr2)):
    b=numpy.reshape(imgarr2[i],(3,32,32))
    cvimg=cv2.merge([b[0],b[1],b[2]])
    graimg=cv2.cvtColor(cvimg,cv2.COLOR_BGR2GRAY)
    descriptionset2.append(vocabulary2.getSIFT(graimg))
descriptionset3=[]
for i in range(len(imgarr3)):
    b=numpy.reshape(imgarr3[i],(3,32,32))
    cvimg=cv2.merge([b[0],b[1],b[2]])
    graimg=cv2.cvtColor(cvimg,cv2.COLOR_BGR2GRAY)
    descriptionset3.append(vocabulary3.getSIFT(graimg))
descriptionset4=[]
for i in range(len(imgarr4)):
    b=numpy.reshape(imgarr4[i],(3,32,32))
    cvimg=cv2.merge([b[0],b[1],b[2]])
    graimg=cv2.cvtColor(cvimg,cv2.COLOR_BGR2GRAY)
    descriptionset4.append(vocabulary4.getSIFT(graimg))
descriptionset5=[]
for i in range(len(imgarr5)):
    b=numpy.reshape(imgarr5[i],(3,32,32))
    cvimg=cv2.merge([b[0],b[1],b[2]])
    graimg=cv2.cvtColor(cvimg,cv2.COLOR_BGR2GRAY)
    descriptionset5.append(vocabulary5.getSIFT(graimg))
descriptionset6=[]
for i in range(len(imgarr6)):
    b=numpy.reshape(imgarr6[i],(3,32,32))
    cvimg=cv2.merge([b[0],b[1],b[2]])
    graimg=cv2.cvtColor(cvimg,cv2.COLOR_BGR2GRAY)
    descriptionset6.append(vocabulary6.getSIFT(graimg))
descriptionset7=[]
for i in range(len(imgarr7)):
    b=numpy.reshape(imgarr7[i],(3,32,32))
    cvimg=cv2.merge([b[0],b[1],b[2]])
    graimg=cv2.cvtColor(cvimg,cv2.COLOR_BGR2GRAY)
    descriptionset7.append(vocabulary7.getSIFT(graimg))
descriptionset8=[]
for i in range(len(imgarr8)):
    b=numpy.reshape(imgarr8[i],(3,32,32))
    cvimg=cv2.merge([b[0],b[1],b[2]])
    graimg=cv2.cvtColor(cvimg,cv2.COLOR_BGR2GRAY)
    descriptionset8.append(vocabulary8.getSIFT(graimg))
descriptionset9=[]
for i in range(len(imgarr9)):
    b=numpy.reshape(imgarr9[i],(3,32,32))
    cvimg=cv2.merge([b[0],b[1],b[2]])
    graimg=cv2.cvtColor(cvimg,cv2.COLOR_BGR2GRAY)
    descriptionset9.append(vocabulary9.getSIFT(graimg))  

vocabularyset=[]

vocabulary0.train(descriptionset0,50)
vocabularyset.append(vocabulary0.vocabulary)

vocabulary1.train(descriptionset1,50)
vocabularyset.append(vocabulary1.vocabulary)

vocabulary2.train(descriptionset2,50)
vocabularyset.append(vocabulary2.vocabulary)
vocabulary3.train(descriptionset3,50)
vocabularyset.append(vocabulary3.vocabulary)
vocabulary4.train(descriptionset4,50)
vocabularyset.append(vocabulary4.vocabulary)
vocabulary5.train(descriptionset5,50)
vocabularyset.append(vocabulary5.vocabulary)
vocabulary6.train(descriptionset6,50)
vocabularyset.append(vocabulary6.vocabulary)
vocabulary7.train(descriptionset7,50)
vocabularyset.append(vocabulary7.vocabulary)
vocabulary8.train(descriptionset8,50)
vocabularyset.append(vocabulary8.vocabulary)
vocabulary9.train(descriptionset9,50)
vocabularyset.append(vocabulary9.vocabulary)

'''
 to Knn
'''
class BovwKnn(object):
    def __init__(self):
        self.vocabularyset=[]
    def classfy(self,img,knum):
        vocabulary=Vocabulary.Vocabulary('getSift')
        descriptions=vocabulary.getSIFT(img)
        vocnum=len(self.vocabularyset)
        

       
 



