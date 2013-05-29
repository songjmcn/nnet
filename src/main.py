'''
Created on 2013-5-28

@author: songjm
'''
import cv2
import cPickle
import gzip
import numpy as np
import os
import nnet
from nnet.softmax import softmax
'''
def testAutoEncoder(path='d:/data/mnist.pkl.gz'):
    net=nnet.Autoencoder(n_in=28*28,n_hidden=100)
    files=os.listdir(path)
    imgs=[]
    for file in files:
        file_path='%s/%s'%(path,file)
        img=cv2.imread(file_path,0)
        img=img.reshape((28*28,))
        imgs.append(img)
    imgs=np.asarray(imgs)
    net.firstRun(imgs)
'''
def testSoftmax(path='d:/data/mnist.pkl.gz'):
    net=softmax(28*28,10)
    zip=gzip.open(path)
    train_set,valid_set,test_set=cPickle.load(zip)
    net.fastTrain(train_set[0][1:100], train_set[1][1:100])
    error=net.Test(valid_set[0], valid_set[1])
if __name__=='__main__':
    testSoftmax()