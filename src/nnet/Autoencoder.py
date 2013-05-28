# encoding: utf-8
'''
nnet.Autoencoder -- 自动编码机

此类是自动编码机的实现

@author:     宋健明
        
@copyright:  2013 天津理工大学. All rights reserved.
        
@license:    GPL

@contact:    songjm1177@gmail.com
@deffield    updated: Updated
'''
import numpy as np
import scipy
from layer import Layer
import active
class AutoEncoder:
    def __init__(self,n_in,n_hidden):
        w=np.random.random_sample(n_hidden,n_in)
        b=np.zeros((n_hidden,1))
        self.hidden=Layer(w,b)
        w=np.random.random_sample(n_in,n_hidden)
        b=np.zeros((n_in,1))
        self.out=Layer(w,b)
        self.beta=3                   
        self.lmb=0.001
        self.sparsityParam=0.05
        self.active=active.sigmoid()
        self.eta=0.01
        return
    def train(self,datas):
        return
    def fastRun(self,data):
        '''
        fast train this net
        '''
        sparsityParam=self.sparsityParam
        lmb=self.lmb
        beta=self.beta
        data=np.transpose(data)
        eta=self.eta
        Jconst=0
        Jweight=0
        Jsparse=0
        n,m=data.shape 
        W1=self.hidden.w
        b1=self.hidden.b
        W2=self.out.w
        b2=self.out.b
        z2=scipy.dot(W1,data)+np.repeat(b1, m, 1)
        a2=self.active.sigmoid(z2)
        z3=scipy.dot(W2,a2)+np.repeat(b2,m,1)
        a3=self.active.sigmoid(z3)
        Jcost=(0.5/m)*np.sum(np.sum(a3-data,axis=1)**2)
        Jweight=(1/2)*(np.sum(np.sum(W1**2,aixs=1))+np.sum(np.sum(W2**2,aixs=1)))
        rho=(1.0/m)*np.sum(a2,aixs=1)
        Jsparse=np.sum(sparsityParam*np.log(sparsityParam/rho),aixs=1)+(1-sparsityParam)*np.log((1-sparsityParam)/(1-rho))
        cost=Jcost+lmb*Jweight+beta*Jsparse
        d3=-(data-a3)*self.active.dsigmoid(z3)
        sterm=beta*(-sparsityParam/rho+(1-sparsityParam)/(1-rho))
        d2=scipy.dot(np.transpose(W1),d3)+np.repeat(sterm,m,1)*self.active.dsigmoid(z2)
        
        W1grad=np.zeros(W1.shape)
        W1grad=W1grad+np.dot(d2,data.transpose())
        W1grad=(1.0/m)*W1grad+lmb*W1
        
        W2grad=np.zeros(W1.shape)
        W2grad=W2grad+scipy.dot(d3,a2.transpose())
        W2grad=(1.0/m)*W2grad+lmb*W2
        b1grad=np.zeros(b1.shape)
        b1grad=b1grad+np.sum(d2,aixs=1)
        b1grad=(1.0/m)*b1grad
        b2grad=np.zeros(b2.shape)
        b2grad=b2grad+sum(d3,aixs=1)
        b2grad=(1.0/m)*b2grad
        
        W1=W1-eta*W1grad
        b1=b1-eta*b1grad
        W2=W2-eta*W2grad
        b2=b2-eta*b2grad
        self.hidden.w=W1
        self.hidden.b=b1
        self.out.w=W2
        self.out.b=b2