# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 11:13:55 2018

@author: Kiruthika
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 12:24:02 2018

@author: Kiruthika
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2

def lsbExtract(stego,noOfLsbBits):
    noOfImageBits = 8
    payload = np.mod(np.left_shift(stego,noOfImageBits-noOfLsbBits),2**noOfImageBits)
    coverMaskBin = np.array(np.concatenate((np.ones(noOfImageBits-noOfLsbBits),np.zeros(noOfLsbBits))),dtype = np.bool)
    coverMask = np.packbits(coverMaskBin)
    cover = np.bitwise_and(stego,coverMask)
    return cover,payload
def getBit(number,position):
    return np.bitwise_and(np.right_shift(number,position-1),1)
    
def extraction(stegoImage,n,x,y,displayImages = False):    
    size = stegoImage.shape
    cover = np.zeros_like(stegoImage)
    payload = np.zeros_like(stegoImage)
    
    for k in range(0,size[2]):
        for i in range(0,size[0]):
#            print(i,'----------------------')
            for j in range(0,size[1],n):
#                print(j)
                if(j+n<=size[1]):
                    cover[i,j,k],status = lsbExtract(stegoImage[i,j,k],n-1)
                    for m in range(1,n):
                        if(getBit(status,m) == 0):
                            cover[i,j+m,k],payload[i,np.int64(j*(n-1)/n+m-1),k] = lsbExtract(stegoImage[i,j+m,k],x)
                        else:
                            cover[i,j+m,k],payload[i,np.int64(j*(n-1)/n+m-1),k] = lsbExtract(stegoImage[i,j+m,k],y)
                    
                        
                    
    if(displayImages):
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(1,3,1)
        ax.imshow(stegoImage)
        ax.set_title('Stego-Image')
        ax = plt.subplot(1,3,2)
        ax.imshow(cover)
        ax.set_title('Cover Image')
        ax = plt.subplot(1,3,3)
        ax.imshow(payload)
        ax.set_title('Payload Image')
        plt.show();
    return cover,payload

def preprocessing(image,size = (1080,720)):
    image = cv2.resize(image,size)
    image = np.array(image,dtype = np.uint8)
    return image
    
n = 3
x = 2
y = 4
size = (1080,720)
stegoImage = preprocessing(cv2.imread('./images/stegoImageEdge53.png'),size)
coverImage,payloadImage = extraction(stegoImage,n,x,y,True)