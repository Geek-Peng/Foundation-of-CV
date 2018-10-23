


import cv2
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import numpy
import scipy.cluster.vq  
file1='F:\\数据集\\cifar-10-python\\cifar-10-batches-py\\data_batch_1'

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

def getSIFT(grayimg):
    sift=cv2.xfeatures2d.SIFT_create()
    
    keypoints,descriptions=sift.detectAndCompute(grayimg,None)
    return descriptions
    
class Vocabulary(object):
    def __init__(self,name):  
        self.name=name
        self.vocabulary=[]  
        self.idf=[]
        self.traindata=[]
        self.num_words=0
        
    def train(self,featureslist,knum):
        num_img=len(featureslist)
        linerlist=[]    
        for i in range(num_img):
            m,n=numpy.shape(featureslist[i])
            for j in range(m):            
                linerlist.append(featureslist[i][j])
        
        linerarray=numpy.array(linerlist)
        wlinerarray=scipy.cluster.vq.whiten(linerarray)  
        
        self.vocabulary,distortion=scipy.cluster.vq.kmeans(wlinerarray,knum,10) 
        self.num_words=numpy.shape(self.vocabulary)[0] 
        print('词汇大小:',numpy.shape(self.vocabulary))
        #遍历所有图像，映射到词汇上
        imwords=numpy.zeros((num_img,self.num_words))
        for i in range(num_img):
            imwords[i]=self.getimhist(featureslist[i])
        print('词频变量格式：',numpy.shape(imwords))
        num_occurences=numpy.sum((imwords>0)*1,axis=0)
        self.idf=numpy.log((1.0*num_img)/(1.0*num_occurences+1))
        self.traindata=featureslist
        return imwords
        
    def getimhist(self,descriptors):
         imhist=numpy.zeros(self.num_words)
         a=numpy.array(descriptors)
         words,distance=scipy.cluster.vq.vq(a,self.vocabulary) 
         for w in words:
             imhist[w]+=1;
         return imhist
         
        
        
        
        
dic=unpickle(file1)
imgarray=dic.get('data')
dataset=imgarray[range(100)]

descriptionset=[] #存储描述子的list
for  i in range(100):
    b=numpy.reshape(dataset[i],(3,32,32))
    cvimg=cv2.merge([b[0],b[1],b[2]])
    grayimg=cv2.cvtColor(cvimg,cv2.COLOR_BGR2GRAY)
    descriptions= getSIFT(grayimg)
    descriptionset.append(descriptions)

vocabulary=Vocabulary('first100vocabulary')
imwordsdistribution=vocabulary.train(descriptionset,500)
row,columns=numpy.shape(imwordsdistribution)
showdata=numpy.zeros(columns)
for j in range(columns):
    for i in range(row):
        showdata[j]+=imwordsdistribution[i][j]

plt.figure(1)
plt.title("distribution of imagewords")
plt.xlabel("词汇")
plt.ylabel("频率")
plt.bar(range(numpy.size(showdata)),showdata,0.001)
plt.show()

'''
BOVW模型进一步工作:
可以选取cifar中几类图片进行训练，然后给出一个新图片，让机器去分类

'''


