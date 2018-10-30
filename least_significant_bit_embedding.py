# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 10:37:13 2018

@author: Kiruthika
"""
import numpy as np
from matplotlib import pyplot as plt
import cv2
def embedding(coverImage,payloadImage,noOfReplaceBits,displayImages = False):
    noOfImageBits = 8
    payload = np.right_shift(payloadImage,(noOfImageBits-noOfReplaceBits))
    coverMaskBin = np.array(np.concatenate((np.ones(noOfImageBits-noOfReplaceBits),np.zeros(noOfReplaceBits))),dtype = np.bool)
    coverMask = np.packbits(coverMaskBin)
    cover = np.bitwise_and(coverImage,coverMask)
    stegoImage = np.bitwise_or(cover,payload)
    if(displayImages):
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(2,3,1)
        ax.imshow(coverImage)
        ax.set_title('Cover Image')
        ax = plt.subplot(2,3,4)
        ax.imshow(cover)
        ax.set_title(str(noOfImageBits-noOfReplaceBits)+' MSBs of Cover Image')
        ax = plt.subplot(2,3,2)
        ax.imshow(payloadImage)
        ax.set_title('Payload Image')
        ax = plt.subplot(2,3,5)
        ax.imshow(payload*2**(noOfImageBits-noOfReplaceBits))
        ax.set_title(str(noOfReplaceBits)+' MSBs of Cover Image')
        ax = plt.subplot(2,3,3)
        ax.imshow(stegoImage)
        ax.set_title('Stego-Image')
        plt.show();
    return stegoImage

def preprocessing(image):
    size = (1080,720)
    image = cv2.resize(image,size)
    image = np.array(image,dtype = np.uint8)
    return image
    


noOfReplaceBits = 3
coverImage = preprocessing(cv2.imread('./images/img5.jpg'))
payloadImage = preprocessing(cv2.imread('./images/img3.png'))
stegoImage = embedding(coverImage,payloadImage,noOfReplaceBits,True)
cv2.imwrite('./images/stegoImage53.png',stegoImage)
