#coding=utf-8
'''
Created on 2013-5-16
@author: 宋健明
'''
import scipy
class tanh:
    def sigmoid(self,data):
        return scipy.tanh(data)
    def dsigmoid(self,data):
        return 1/scipy.cosh(data)
class sigmoid:
    def __init__(self,a=1):
        self.a=a;
    def sigmoid(self,data):
        return 1.0/(1+scipy.exp(-self.a*data))
    def dsigmoid(self,data):
        return (self.a*scipy.exp((0-self.a)*data)/scipy.power((1+scipy.exp((0-self.a)*data)), 2))