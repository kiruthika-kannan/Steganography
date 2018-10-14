# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 12:24:02 2018

@author: Kiruthika
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
def extraction(stegoImage,noOfReplaceBits,displayImages = False):
    noOfImageBits = 8
    payload = np.mod(np.left_shift(stegoImage,noOfImageBits-noOfReplaceBits),2**noOfImageBits)
    print(payload)
    coverMaskBin = np.array(np.concatenate((np.ones(noOfImageBits-noOfReplaceBits),np.zeros(noOfReplaceBits))),dtype = np.bool)
    coverMask = np.packbits(coverMaskBin)
    cover = np.bitwise_and(stegoImage,coverMask)
    if(displayImages):
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(1,3,1)
        ax.imshow(stegoImage)
        ax.set_title('Stego-Image')
        ax = plt.subplot(1,3,2)
        ax.imshow(cover)
        ax.set_title(str(noOfImageBits-noOfReplaceBits)+' MSBs of Cover Image')
        ax = plt.subplot(1,3,3)
        ax.imshow(payload)
        ax.set_title(str(noOfReplaceBits)+' MSBs of Cover Image')
        plt.show();
    return cover,payload

def preprocessing(image):
    size = (1080,720)
    image = cv2.resize(image,size)
    image = np.array(image,dtype = np.uint8)
    plt.imshow(image)
    plt.show()
    return image
    
noOfReplaceBits = 4
stegoImage = preprocessing(cv2.imread('./images/stegoImage.png'))
coverImage,payloadImage = extraction(stegoImage,noOfReplaceBits,True)