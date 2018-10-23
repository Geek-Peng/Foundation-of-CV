# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:41:03 2017

@author: God鹏
BOVW生成图像词汇的类
"""
import cv2
import numpy
import scipy.cluster.vq
class Vocabulary(object):
    def __init__(self,name):
        self.name=name
        self.vocabulary=[]
    def getSIFT(self,grayimg):
        sift=cv2.xfeatures2d.SIFT_create()
        keypoints,descriptions=sift.detectAndCompute(grayimg,None)
        return descriptions
        
        
    def train(self,descriptions,knum):  
        img_num=len(descriptions)
        linerlist=[]
        for i in range(img_num):
           
            descriptions[i]
            
            if not descriptions[i] is None:  
                m,n=numpy.shape(descriptions[i])
                for j in range(m):            
                    linerlist.append(descriptions[i][j])
            
        linerarray=numpy.array(linerlist)
        wlinerarray=scipy.cluster.vq.whiten(linerarray) #白化
        self.vocabulary,distortion=scipy.cluster.vq.kmeans(wlinerarray,knum,10) #进行10轮聚类
    
