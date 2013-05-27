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
class Autoencoder:
    def __init__(self,layers):
        '''
        初始化自动编码机的各层
        layer是各层神经元的格式，共有三层：输入层 隐藏层 输出层(输入层的神经元个数等于输出层的个数)
        '''
        if len(layers)!=3:
            raise TypeError('layers must be s tuple and have 3 elements')
        n_in=layers[0]
        n_out=layers[1]
        fanin=n_in*n_out
        sd=1.0/scipy.sqrt(fanin)
        w=sd*np.random.random_sample((n_in,n_out))
        b=np.zeros(n_out)
        self.hidden=Layer(w,b)
        n_in=layers[1]
        n_out=layers[2]
        fanin=n_in*n_out
        sd=1.0/scipy.sqrt(fanin)
        w=sd*np.random((n_in,n_out))
        b=np.zeros(n_out)
        self.out=Layer(w,b)
        self.active=active.sigmoid()
        self.eta=0.01
        self.bta=0.1
        return
    def forward(self,sample):
        """
        前向传播
        sample  训练样本
        """
        sample=np.asarray(sample,dtype='float32')
        hidden_output=self.comput(sample, self.hidden.w, self.hidden.b);
        out_output=self.comput(hidden_output, self.out.w, self.out.b);
        return hidden_output,out_output
    def backward(self,input,p,hidden_output,out_output):
        '''
        反向传播
        input 网络输入值
        p 算出的P(p=aveg(out))
        hidden_output 隐藏层输出
        out_output 输出层输出
        '''
        dout=(p-out_output)
        dout1,dw2,db2=self.dlayer(dout, out_output, hidden_output, self.out.w,self.out.b)
        self.out.w=self.out.w-self.eta*dw2
        self.out.b=self.out.b-self.eta*db2
        din1,dw1,db1=self.dlayer(dout1, hidden_output, input, self.hidden.w, self.hidden.b)
        self.hidden.w=self.hidden.w-self.eta*dw1
        self.hidden.b=self.hidden.b-self.eta*db1
    def comput(self,input,w,b):
        return self.active.sigmoid(scipy.dot(input,w)+b)
    def dlayerout(self,dout,out,input,w,b):
        dout=dout*self.active.dsigmoid(out)
        loop=b.shape[0]
        din=np.zeros(input.shape)
        dw=np.zeros(w.shape)
        db=np.zeros(b.shape)
        for ii in xrange(loop):
            this_dout=dout[ii]
            this_w=w[:,ii]
            b[ii]=this_dout
            this_dw=input*this_dout
            this_din=this_dout*this_w
            dw[:,ii]=this_dw
            din+=this_din
        return din,dw,db
    def dlayer_hidden(self,dout,out,input,w,b,p):
        dw=np.zeros(w.shape)
        db=np.zeros(b.shape)
        loop=b.shape[0]
        d=self.active.dsigmoid(out)
        for ii in xrange(loop):
            dout[ii]=dout[ii]+self.bta*(-(input[ii]/p[ii])+(1-input[ii])/(1-p[ii]))
        dout=dout*self.active.dsigmod(out)
        din=np.zeros(input.shape)
        dw=np.zeros(w.shape)
        db=np.zeros(b.shape)
        for ii in xrange(loop):
            this_dout=dout[ii]
            b[ii]=this_dout
            this_dw=input*this_dout
            dw[:,ii]=this_dw
        return dw,db
    def run(self,data):
        hout=self.comput(data, self.hidden.w, self.hidden.b)
        out=self.comput(hout, self.out.w, self.out.b)
        return out
class FastAutoencoder:
    def __init__(self,n_in,n_hidden):
        self.encoder=Autoencoder((n_in,n_hidden,n_in))
        self.input=[]
        self.hidden_output=[]
        self.out_output=[]
        self.p=np.zeros((n_in,),dtype='float32')
        return
    def train(self,samples):
        loop=len(samples)
        for ii in xrange(loop):
            s=samples[ii]
            hout,out=self.encoder(s)
            self.hidden_output.append(hout)
            self.out_output.append(out)
            self.input.append(s)
            self.p+=out
        self.p=self.p/loop
        for ii in xrange(loop):
            hout=self.hidden_output[ii]
            out=self.out_output[ii]
            input=self.input[ii]
            self.encoder.backward(input, self.p, hout, out)
        return
    def comput(self,data):
        return self.encoder.run(data)
            