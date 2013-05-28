#coding=utf-8
'''
Created on 2013-5-16
@author: 宋健明
'''
import scipy
from scipy import tanh
from scipy import cosh
from scipy import exp
class tanh:
    def sigmoid(self,data):
        return tanh(data)
    def dsigmoid(self,data):
        return 1.0/cosh(data)
class sigmoid:
    def sigmoid(self,data):
        return 1.0 /(1+exp(-data))
    def dsigmoid(self,data):
        return self.sigmoid(data)*(1-self.sigmoid(data))