#coding=utf-8
'''
Created on 2013-5-28

@author: songjm
SoftMax 回归
'''
from numpy import *
from active import *
class softmax:
    def __init__(self,n_in,n_out):
        fanin=n_in*n_out
        sd=1.0/sqrt(fanin)
        self.w=sd*random.random_sample((n_in,n_out))
        self.b=zeros((1,n_out))
        self.eta=0.01
        self.Lambda=exp(-4)
        return
    def fastTrain(self,samples,labels):
        w=self.w
        b=self.b
        m=samples.shape[0]
        theta=dot(samples,w)+repeat(b,m,0)
        out=Softmax(theta)
        targetOut=zeros(out.shape)
        targetOut[[array(range(m)),labels]]=1
        dout=-(targetOut-out)
        wgrad=1.0/m*dot(transpose(samples),dout)+self.Lambda*w
        w=w-self.eta*wgrad
        b=1.0/m*sum(dout,0)
        self.w=w
        self.b=b-self.eta*b.reshape(self.b.shape)
        return
    def Test(self,samples,labels):
        w=self.w
        b=self.b
        m=samples.shape[0]
        theta=dot(samples,w)
        out=Softmax(theta)
        pre_labels=argmax(out, 1)
        result=mean(pre_labels!=labels)
        return result
def Softmax(theta):
    '''
    softmax 回归
    '''
    n=theta.shape[1]
    tmp=exp(theta)
    s=array([sum(tmp,1)])
    s=transpose(s)
    s=repeat(s,n,1)
    result=tmp/s
    return result
class HiddenLayer:
    def __init__(self,n_in,n_out,active=sigmoid(),eta=0.01):
        fanin=n_in*n_out
        sd=1.0/sqrt(fanin)
        self.w=sd*random.random_sample((n_in,n_out))
        self.b=zeros((1,n_out))
        self.active=active
        self.out=None
        self.Lambda=exp(-4)
        self.eta=eta
        return
    def forward_fast(self,samples):
        w=self.w
        b=self.b
        active=self.active
        m=samples.shape[0]
        out=active.sigmoid(dot(samples,w)+repeat(b,m,0))
        self.out=out
        return out
    def backward_fast(self,samples,dout):
        m=samples.shape[0]
        w=self.w
        b=self.b
        active=self.active
        out=self.out
        Lambda=self.Lambda
        eta=self.eta
        dout=dout*active.dsigmoid(out)
        dw=1.0/m*dot(transpose(samples),dout)+Lambda*w
        db=1.0/m*sum(dout,0)
        din=1.0/m*dot(dout,transpose(w))
        self.w=w-eta*dw
        self.b=b-eta*db.reshape(b.shape)
        return din
class SoftmaxLayer:
    def __init__(self,n_in,n_out,eta):
        fanin=n_in*n_out
        sd=1.0/sqrt(fanin)
        self.w=sd*random.random_sample((n_in,n_out))
        self.b=zeros((1,n_out))
        self.eta=eta
        self.Lambda=exp(-4)
        return
    def forward_fast(self,samples):
        m=samples.shape[0]
        w=self.w
        b=self.b
        out=dot(samples,w)+repeat(b,m,0)
        h=Softmax(out)
        return h
    def backward_fast(self,samples,dout):
        w=self.w
        b=self.b
        eta=self.eta
        Lambda=self.Lambda
        m=samples.shape[0]
        dw=1.0/m*dot(transpose(samples),dout)+Lambda*w
        db=1.0/m*sum(dout,0)
        db=reshape(db,b.shape)
        din=dot(dout,transpose(w))
        self.w=w-eta*dw
        self.b=b-eta*db
        return din
class Backpropagation:
    def __init__(self,n_in,n_hidden,n_out,eta):
        '''
        初始化网络，此网络的输出层为softmax回归
        n_layers  每一层的神经元数
        '''
        self.hidden=HiddenLayer(n_in,n_hidden,eta=eta)
        self.out=SoftmaxLayer(n_hidden,n_out,eta)
        return
    def fastTrain(self,sample,labels):
        m=sample.shape[0]
        input=sample
        #print('input\n %s'%(input))
        out=self.hidden.forward_fast(input)
       # print('hidden output \n%s'%(out))
        input=out
        h=self.out.forward_fast(input)
        #print('soft max output\n %s'%(h))
        "反向传播"
        targetOut=zeros(h.shape)
        targetOut[[array(range(m)),labels]]=1
        #print('targetOut\n%s'%(targetOut))
        dout=-(targetOut-h)
        dout=self.out.backward_fast(self.hidden.out, dout)
        dout=self.hidden.backward_fast(sample, dout)
    def test(self,samples,labels):
        out=self.hidden.forward_first(samples)
        h=self.out.forward_first(out)
        pre=argmax(h,1)
        return mean(pre!=labels) 