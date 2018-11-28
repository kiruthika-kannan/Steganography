# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 00:25:38 2018

@author: Kiruthika
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
from utils import preprocessing, evaluate

def embedding(coverImage,payloadImage,displayImages = False):
    noOfImageBits = 8
    size = coverImage.shape
    stegoImage = np.zeros_like(coverImage)
    #Calc range and uk,lk
    for i in range(0, size[0]):
        for j in range(0,size[1],2):
            payload_j = np.int(j/2)
            for k in range(0,3):
                d = np.int(coverImage[i,j+1,k])-np.int(coverImage[i,j,k])
                [l,u] = quantize(abs(d))
                noOfReplaceBits = np.int(np.log2(u-l+1))
                p = payloadImage[i,payload_j,k]
                #Calc b = f(p,n)
                b = np.right_shift(p,(noOfImageBits-noOfReplaceBits))
                #Calc new_d =f( +- lk+b ,d)
                if (d>=0):
                    new_d = l + b;
                else:
                    new_d = - (l + b)
                
                m = new_d - d
                if(np.mod(d,2) == 1):
                    stegoImage[i,j,k] = coverImage[i,j,k] - np.ceil(m/2)
                    stegoImage[i,j+1,k] = coverImage[i,j+1,k] + np.floor(m/2)
                else:
                    stegoImage[i,j,k] = coverImage[i,j,k] - np.floor(m/2)
                    stegoImage[i,j+1,k] = coverImage[i,j+1,k] + np.ceil(m/2)
                #Check fallOffBoundary(stegoImage[i,j,k])
                    #stegoImage[i,j,k] = coverImage[i,j,k]
                #Check fallOffBoundary(stegoImage[i,j+1,k])
                    #stegoImage[i,j+1,k] = coverImage[i,j+1,k]
                if(stegoImage[i,j,k] <= 0 or stegoImage[i,j,k] >= 255 or stegoImage[i,j+1,k] <= 0 or stegoImage[i,j+1,k] >= 255 ):
                    stegoImage[i,j,k] = coverImage[i,j,k]
                    stegoImage[i,j+1,k] = coverImage[i,j+1,k]
            

    if(displayImages):
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(1,3,1, aspect='auto')
        ax.imshow(coverImage, aspect='auto')
        ax.set_title('Cover Image')
        ax = plt.subplot(1,3,2, aspect='auto')
        ax.imshow(payloadImage, aspect='auto')
        ax.set_title('Payload Image')
        ax = plt.subplot(1,3,3, aspect='auto')
        ax.imshow(stegoImage, aspect='auto')
        ax.set_title('Stego-Image')
        plt.show();
    return stegoImage



def quantize(d):    
    quantization = np.array([[0,7],[8,15],[16,31],[31,63],[64,127],[128,255]])    
    for i in range(0,quantization.shape[0]):
        if(d<=quantization[i,1]):
            return quantization[i,:]
    return None           

def test():
    coverImage = preprocessing(cv2.imread('./images/img5.jpg'),(1080,720))
    payloadImage = preprocessing(cv2.imread('./images/img3.png'),(540,720))
    stegoImage = embedding(coverImage,payloadImage,True)
    
    cv2.imwrite('./images/coverImagePVD53.png',coverImage)
    cv2.imwrite('./images/stegoImagePVD53.png',stegoImage)
