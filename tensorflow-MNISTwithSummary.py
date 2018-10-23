# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 09:57:14 2018

@author: God鹏
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:25:01 2018

@author: God鹏
"""
'''
histogram是整个训练过程,,用的是summary.histogram
events显示损失函数变化啥的,用的是summary.scalar
'''

import tensorflow as tf
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES 
from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.python.platform import gfile 

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_22(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
mnist=input_data.read_data_sets("MNISTZIP/",one_hot=True)
if gfile.Exists("E:/lmgod/Pythonws/CNNMNISTLOG"):
    gfile.DeleteRecursively("E:/lmgod/Pythonws/CNNMNISTLOG")
sess=tf.InteractiveSession()




with tf.name_scope("input"):
    x=tf.placeholder(tf.float32,[None,784],name="x")
    y_=tf.placeholder(tf.float32,[None,10],name="y")  
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    s1=tf.summary.image('input', image_shaped_input, 10)
with tf.name_scope("conv1"):  
    W_conv1=weight_variable([5,5,1,32])
    s2=tf.summary.histogram("conv1_W",W_conv1)
    b_conv1=bias_variable([32])
    x_image=tf.reshape(x,[-1,28,28,1])
    h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
    h_pool1=max_pool_22(h_conv1)
with tf.name_scope("conv2"):
    W_conv2=weight_variable([5,5,32,64])
    s3=tf.summary.histogram("conv2_W",W_conv2)
    b_conv2=bias_variable([64])
    h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    h_pool2=max_pool_22(h_conv2)
with tf.name_scope("fullconnect"):
    W_fc1=weight_variable([7*7*64,1024])
    b_fc1=bias_variable([1024])
    h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
    h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
with tf.name_scope("DropOut"):
    keep_prob=tf.placeholder("float")
    h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
with tf.name_scope("output"):
    W_fc2=weight_variable([1024,10])
    b_fc2=bias_variable([10])
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
    s4=tf.summary.histogram("output",y_conv)

with tf.name_scope("loss"):
    cross_entropy=-tf.reduce_sum(y_*tf.log(y_conv))
s5=tf.summary.scalar("loss",cross_entropy) #这里的键值必须是个scop的name
with tf.name_scope("train"):
    train_step=tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)
with tf.name_scope("accure"):
    with tf.name_scope("correct_prediction"):
        correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    with tf.name_scope("accure"):        
        accuracy=tf.reduce_mean(tf.cast(correct_prediction , "float"))
s6=tf.summary.scalar("accure",accuracy)

merged=tf.summary.merge([s1,s2,s3,s4,s5,s6]) 

#==============以上网络结构定义完了，下面开始训练====================#
'''
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
      train_accuracy=accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
      print("step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})
      print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
'''

'''
i=0
batch = mnist.train.next_batch(50)
#train_accuracy=accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})#无可视化写法
#logs=sess.run(merged,feed_dict={x:batch[0], y_: batch[1] ,keep_prob: 1.0})这么写会报错InvalidArgumentError: You must feed a value for placeholder tensor 'input/y' with dtype float and shape [?,10]
logs,train_accuracy=sess.run([merged,accuracy],feed_dict={x:batch[0], y_: batch[1] ,keep_prob: 1.0})
writer.add_summary(logs,i)
print("step %d, training accuracy %g"%(i, train_accuracy))

#train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})  #这个写法和下面写法一样，tf每个节点都封装了run函数，或者sess.run某个节点是一样的
sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})
result=accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
print("test accuracy %g"%result)
'''

writer=tf.summary.FileWriter("E:/lmgod/Pythonws/CNNMNISTLOG",sess.graph) 
sess.run(tf.global_variables_initializer())
i=0
while True:
    batch = mnist.train.next_batch(50)
    k=i
    if k%100 == 0:
       
        logs,train_accuracy=sess.run([merged,accuracy],feed_dict={x:batch[0], y_: batch[1] ,keep_prob: 1.0})
        writer.add_summary(logs,i)
        print("step %d, training accuracy %g"%(k, train_accuracy))
        
        sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})
        
    i=i+1
    if train_accuracy>0.9:
        break
'''
BP训练简单网络
'''
'''
mnist=input_data.read_data_sets("MNISTZIP/",one_hot=True)#手动下载了然后不用解压放在跟工程同目录下即可，文件夹叫MNISTZIP
x=tf.placeholder("float",[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b) #模型估计的y,y有10个分量，每个分量为这个图片属于那个类别的可能性
y_=tf.placeholder("float",[None,10])  #这是y的客观真值
cross_entropy=-tf.reduce_sum(y_*tf.log(y))
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuray=tf.reduce_mean(tf.cast(correct_prediction,"float"))
print(sess.run(accuray,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
'''
