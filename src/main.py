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
from nnet.softmax import Backpropagation
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
    bitch_size=500
    net=softmax(28*28,10)
    net.eta=0.02
    zip=gzip.open(path)
    train_set,valid_set,test_set=cPickle.load(zip)
    n_train=train_set[0].shape[0]/bitch_size
    n_test=valid_set[0].shape[0]
    loop=100
    for lp in xrange(loop):
        for ii in xrange(n_train):
            start=ii*bitch_size
            end=(ii+1)*bitch_size
            this_sample=train_set[0][start:end]
            this_label=train_set[1][start:end]
            net.fastTrain(this_sample,this_label)
        error=net.Test(test_set[0], test_set[1])
        print('best error : %s'%(error))
    return
def testBackpropagation(path='d:/data/mnist.pkl.gz'):
    bitch_size=2
    net=Backpropagation(28*28,50,10,0.01)
    zip=gzip.open(path)
    train_set,valid_set,test_set=cPickle.load(zip)
    n_train=train_set[0].shape[0]/bitch_size
    n_test=valid_set[0].shape[0]
    loop=100
    for lp in xrange(loop):
        for ii in xrange(n_train):
            start=ii*bitch_size
            end=(ii+1)*bitch_size
            this_sample=train_set[0][start:end]
            this_label=train_set[1][start:end]
            net.fastTrain(this_sample,this_label)
            return
        error=net.test(test_set[0], test_set[1])
        print('current error :%s'%(error))
if __name__=='__main__':
    testSoftmax()