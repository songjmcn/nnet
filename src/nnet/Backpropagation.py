#coding=utf-8
'''
Created on 2013-5-28

@author: songjm
反向传播神经网络
'''
from numpy import *
from layer import Layer
from active import sigmoid
class Backpropagation:
    def __init__(self,layers):
        if layers==None:
            raise TypeError('layers is none')
        length=len(layers)
        tmp_layers=[]
        for i in xrange(length):
            n_in=layers[i]
            n_out=layers[i+1]
            fanin=n_in*n_out
            sd=1.0/sqrt(fanin)
            w=sd*random.random_sample((n_in,n_out))
            b=zeros((1,n_out))
            this_layer=Layer(w,b)
            tmp_layers.append(this_layer)
        self.layers=tmp_layers
        self.active=sigmoid()
        self.Lambda=exp(-4)
        self.n_out=layers[len(layers)-1]
        self.eta=0.01
        return
    def fastTrain(self,samples,labels):
        '''
          对样本进行训练
          sample  样本 二维矩阵 每一行是一个样本
          labels    样本的标签，一维向量
        '''
        m=samples.shape[0]
        all_out=[]
        all_out.append(samples)
        active=self.active
        input=samples
        '''
        前向传播，计算每一层的输出
        '''
        for layer in self.layers:
            w=layer.w
            b=layer.b
            this_z=dot(input,w)+repeat(b,m,0)   "z=w*x+b"
            this_a=active.sigmoid(this_z)            'a=f(z)'
            input=this_a
            all_out.append(this_a)
        '''
        反向传播，微调每一层的参数
        计算每一层的偏导数，并计算W b的梯度
        对于输出层，输出的偏导数为dout=-(y-a)*df(a) df为激活函数的导数
        对于其它层，输出的偏导数为dout=W*dout*df(a)
        wgrad=a*dout   a为本层的输出值
        bgrad=dout
        '''
        length=-len(self.layers)-1
        Lambda=self.Lambda
        eta=self.eta
        targetOut=zeros((m,self.n_out))
        targetOut[[array(range(m),labels)]]=1
        dout=-(targetOut-all_out[-1])*active.dsigmoid(all_out[-1])   '输出层的偏导数'
        for index in xrange(-1,length-1,-1):
            layer=self.layers[index]
            w=layer.w
            b=layer.b
            input=all_out[index-1]
            input=transpose(input)
            wgrad=1.0/m*dot(input,dout)+Lambda*w
            bgrad=1.0/m*sum(dout,0)
            w=w-eta*wgrad
            b=b-eta*bgrad
            self.layers[index].w=w
            self.Lambda[index].b=b
            tw=transpose(w)
            dout=dot(dout,w)*active.dsigmoid(input)
        return
    def test(self,samples,labels):
        m=samples.shape[0]
        input=samples
        for layer in self.layers:
            w=layer.w
            b=layer.b
            this_z=dot(input,w)+repeat(b,m,0)
            this_a=self.active.sigmoid(this_z)
            input=this_a
        pre_labels=argmax(input,1)
        result=array(not_equal(pre_labels,labels),dtype='int')
        return mean(result)
    def pre(self,samples):
        m=samples.shape[0]
        input=samples
        for layer in self.layers:
            w=layer.w
            b=layer.b
            this_z=dot(input,w)+repeat(b,m,0)
            this_a=self.active.sigmoid(this_z)
            input=this_a
        pre_labels=argmax(input,1)
        return pre_labels