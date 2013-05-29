#coding=utf-8
'''
Created on 2013-5-28

@author: songjm
SoftMax 回归
'''
from numpy import *
class softmax:
    def __init__(self,n_in,n_out):
        self.w=zeros((n_in,n_out))
        self.b=zeros((1,n_out))
        self.eta=0.01
        self.Lambda=exp(-4)
        return
    def fastTrain(self,samples,labels):
        w=self.w
        b=self.b
        m=samples.shape[0]
        theta=dot(w,samples)+repeat(self.b,m,0)
        out=Softmax(theta)
        targetOut=zeros(out.shape)
        targetOut[[array(range(m)),labels]]=1
        dout=-(targetOut-out)
        wgrad=1.0/m*dot(transpose(samples),dout)+self.Lambda*w
        w-=self.eta*wgrad
        b=1.0/m*sum(dout,0)
        self.w=w
        self.b=b
        return
    def Test(self,samples,labels):
        w=self.w
        b=self.b
        m=samples.shape[0]
        theta=dot(w,samples)+repeat(b,m,0)
        out=softmax(theta)
        pre_labels=argmax(out, 1)
        result=mean(pre_labels!=labels)
def Softmax(theta):
    '''
    softmax 回归
    '''
    tmp=exp(theta)
    s=sum(tmp,0)
    return tmp/s