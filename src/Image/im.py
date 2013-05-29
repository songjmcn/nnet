'''
Created on 2013-5-28

@author: songjm
'''
import cv2
import numpy as np
def imrotate(img,degree):
    angle=degree*cv2.CV_PI/180.0
    a=np.sin(angle)
    b=np.cos(angle)
    height,width=img.shape
    width_rotate=int(height*np.fabs(a)+width*np.fabs(b))
    height_rotate=int(width*np.fabs(a)+height*np.fabs(b))
    tempLength=np.sqrt(width)*width
    