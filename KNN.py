# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 15:59:03 2017

@author: God鹏
"""


import numpy

class KnnClassifier(object):
    def __init__(self,lable,samples):
        self.lable=lable
        self.samples=samples
        
        
    def classify(self,point,k):
        distances=[]
        for i in range(len(self.samples)):
            distances.append(self.getdistance(point,self.samples[i]))
        distances=numpy.array(distances)
        votes={}
        order=numpy.argsort(distances)   
        for kk in range(k):      
           klable=self.lable[order[kk]] 
           votes.setdefault(klable,0) 
           votes[klable]+=1
        return max(votes)
                
        
    def getdistance(self,point1,point2):
        distance=numpy.sqrt(numpy.sum((point1-point2)**2))
        return distance
            



class1=0.6*numpy.random.randn(200,2)
class2=1.2*numpy.random.randn(200,2)+5
lable1=[]
lable2=[]
for i in range(200):
    lable1.append('class1')
    lable2.append('class2')
samples = numpy.vstack((class1,class2)) 
lables = numpy.hstack((lable1,lable2))

knn=KnnClassifier(lables,samples)
testpoint=numpy.float64([0,1])
result=knn.classify(testpoint,5)
print(result)



