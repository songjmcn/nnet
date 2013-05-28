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
def testAutoEncoder(path='d:/data/mnist'):
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
if __name__=='__main__':
    testAutoEncoder()