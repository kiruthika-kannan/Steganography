# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 01:33:25 2018

@author: Kiruthika
"""


import numpy as np
from matplotlib import pyplot as plt
import cv2
from utils import preprocessing, evaluate

def max_bpcs_complexity(ncols, nrows):
    return ((nrows-1)*ncols) + ((ncols-1)*nrows)

def arr_bpcs_complexity(arr):
    nrows, ncols = arr.shape
    max_complexity = max_bpcs_complexity(nrows, ncols)
    nbit_changes = lambda items, length: sum([items[i] ^ items[i-1] for i in range(1, length)])
    k = 0
    for row in arr:
        k += nbit_changes(row, ncols)
    for col in arr.transpose():
        k += nbit_changes(col, nrows)
    return (k*1.0)/max_complexity

def checkerboard(h, w):
    re = np.r_[ np.int(w/2)*[0,1] + ([0] if w%2 else [])]
    ro = 1-re
    return np.row_stack(np.int(h/2)*(re,ro) + ((re,) if h%2 else ()))

def conjugate(arr):
    wc = checkerboard(arr.shape[0], arr.shape[1]) # white pixel at origin
    bc = np.uint8(1-wc) # black pixel at origin
    return np.bitwise_xor(arr,bc)


def lsbExtract(stego,noOfLsbBits):
    noOfImageBits = 8
    payload = np.mod(np.left_shift(stego,noOfImageBits-noOfLsbBits),2**noOfImageBits)
    coverMaskBin = np.array(np.concatenate((np.ones(noOfImageBits-noOfLsbBits),np.zeros(noOfLsbBits))),dtype = np.bool)
    coverMask = np.packbits(coverMaskBin)
    cover = np.bitwise_and(stego,coverMask)
    return cover,payload

def getBit(number,position):
    return np.bitwise_and(np.right_shift(number,position-1),1)
    
def complexity(img):
    alpha = []
    for i in range(0,8):
        bitPlaneMask = np.uint8(2**(i))
        plane = np.bitwise_and(img,bitPlaneMask)
        plane = np.uint8(plane/np.max(plane))
        a = arr_bpcs_complexity(plane)
        alpha.append(a)
        
    return alpha

def complexityConjugateFromMap(img,cMap,noOfReplaceBits):
    noOfImageBits = 8
    conjugateMap = np.unpackbits(np.uint8(cMap))
    for i in range(0,noOfReplaceBits):
        if (conjugateMap[i] == 1):
            bitPlaneMask = np.uint8(2**(noOfImageBits-i-1))
            plane = np.bitwise_and(img,bitPlaneMask)
            if(plane.max()>1):
                plane = np.uint8(plane/plane.max())
            conjugatePlane = conjugate(plane)*bitPlaneMask
            img = np.bitwise_and(img,np.bitwise_not(bitPlaneMask))
            img = np.bitwise_or(img,conjugatePlane)
            
    return img
    
def extraction(stegoImage,n,x,y,displayImages = False):    
    size = stegoImage.shape
    alphaThreshold = 0.3
    cover = np.zeros_like(stegoImage)
    payload = np.zeros_like(stegoImage)
    
    rtemp = np.ones_like(stegoImage)
    
    for k in range(0,size[2]):
        for i in range(0,size[0],n):
            if(i+n<=size[0]):
#                print(i,':',k)
                for j in range(0,size[1],n):
    #                print(j)
                    if(j+n<=size[1]):
                        cover[i,j,k],conjugateMap = lsbExtract(stegoImage[i,j,k],y) 
                        alphaStego = complexity(stegoImage[i:i+n,j:j+n,k])
                        noOfReplaceBits = x
                        for m in range(x+1,y):
                            if(alphaStego[m]>alphaThreshold):
                                noOfReplaceBits = m
                        cover[i:i+n,j:j+n,k],payld = lsbExtract(stegoImage[i:i+n,j:j+n,k],noOfReplaceBits)
                        payload[i:i+n,j:j+n,k] = complexityConjugateFromMap(payld, conjugateMap,noOfReplaceBits )
                        payload[i,j,k] = np.average([payload[i+1,j+1,k], payload[i,j+1,k], payload[i+1,j,k]])
                        rtemp[i:i+n,j:j+n,k] = noOfReplaceBits*rtemp[i:i+n,j:j+n,k]
                                
                    
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

    
def test():
    n = 8
    x = 3
    y = 6
    
    size = (1080,720)
    #size = (80,80)
    stegoImage = preprocessing(cv2.imread('./images/stegoImageBPCS53.png'),size)
    coverImage,payloadImage = extraction(stegoImage,n,x,y,True)