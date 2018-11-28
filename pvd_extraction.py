# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 01:04:58 2018

@author: Kiruthika
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
from utils import preprocessing, evaluate

def extraction(stegoImage,displayImages = False):
    noOfImageBits = 8
    size = stegoImage.shape
    payloadSize = list(size)
    payloadSize[1] = np.int(payloadSize[1]/2)
    cover = np.zeros_like(stegoImage)
    payload = np.zeros_like(stegoImage)
    payload = preprocessing(payload,tuple(payloadSize))
    #Calc range and uk,lk
    for i in range(0, size[0]):
        for j in range(0,size[1],2):
            payload_j = np.int(j/2)
            for k in range(0,3):                
                d = np.int(stegoImage[i,j+1,k])-np.int(stegoImage[i,j,k])
                [l,u] = quantize(abs(d))
                #g1 = coverImage[i,j,k] - f(floor/ceil(m/2) , uk - d)
                #g2 = coverImage[i,j+1,k] + f(floor/ceil(m/2) , uk - d)
                m = u - d
                if(np.mod(d,2) == 1):
                    cover[i,j,k] = stegoImage[i,j,k] - np.ceil(m/2)
                    cover[i,j+1,k] = stegoImage[i,j+1,k] + np.floor(m/2)
                else:
                    cover[i,j,k] = stegoImage[i,j,k] - np.floor(m/2)
                    cover[i,j+1,k] = stegoImage[i,j+1,k] + np.ceil(m/2)
                    
                if(cover[i,j,k] <= 0  or cover[i,j,k] >= 255 or cover[i,j+1,k] <= 0 or cover[i,j+1,k] >= 255 ):
                    cover[i,j,k] = stegoImage[i,j,k]
                    cover[i,j+1,k] = stegoImage[i,j+1,k]
                else:
                    #Calculate n = log2(uk-lk+1)
                    noOfReplaceBits = np.int(np.log2(u-l+1))
#                    b = f(+/- d -lk,d)
                    if (d>=0):
                        b = d - l
                    else:
                        b = - d - l
                    payload[i,payload_j,k] = np.mod(np.left_shift(b,noOfImageBits-noOfReplaceBits),2**noOfImageBits)
            
    if(displayImages):
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(1,3,1)
        ax.imshow(stegoImage, aspect='auto')
        ax.set_title('Stego-Image')
        ax = plt.subplot(1,3,2, aspect='auto')
        ax.imshow(cover, aspect='auto')
        ax.set_title('Cover Image')
        ax = plt.subplot(1,3,3, aspect='auto')
        ax.imshow(payload, aspect='auto')
        ax.set_title('Payload Image')
        plt.show();
    return cover,payload


def quantize(d):    
    quantization = np.array([[0,7],[8,15],[16,31],[31,63],[64,127],[128,255]])    
    for i in range(0,quantization.shape[0]):
        if(d<=quantization[i,1]):
            return quantization[i,:]
    return None              

def test():
    stegoImage = preprocessing(cv2.imread('./images/stegoImagePVD53.png'))
    coverImage,payloadImage = extraction(stegoImage,True)

