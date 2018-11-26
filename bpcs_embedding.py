# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 19:34:57 2018

@author: Kiruthika
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 01:48:25 2018

@author: Kiruthika
"""


import numpy as np
from matplotlib import pyplot as plt
import cv2
def max_bpcs_complexity(ncols, nrows):
    return ((nrows-1)*ncols) + ((ncols-1)*nrows)

def arr_bpcs_complexity(arr):
    """
    arr is a 2-d numpy array
    returns the fraction of maximum bpcs_complexity in arr
        where bpcs_complexity is total sum of bit changes
        moving along each row and each column
    """
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
    """
    h, w are int
    returns a checkerboard array of shape == [h,w]
    """

    re = np.r_[ np.int(w/2)*[0,1] + ([0] if w%2 else [])]
    ro = 1-re
    return np.row_stack(np.int(h/2)*(re,ro) + ((re,) if h%2 else ()))

def conjugate(arr):
    """
    arr is a numpy array

    conjugates arr so that its complexity, s, is 1-s
    assert conjugate(conjugate(arr)) == arr
    """
    wc = checkerboard(arr.shape[0], arr.shape[1]) # white pixel at origin
    bc = np.uint8(1-wc) # black pixel at origin
    return np.bitwise_xor(arr,bc)


def lsbEmbed(cover,payload,noOfReplaceBits):
    noOfImageBits = 8
    payload = np.right_shift(payload,(noOfImageBits-noOfReplaceBits))
    coverMaskBin = np.array(np.concatenate((np.ones(noOfImageBits-noOfReplaceBits),np.zeros(noOfReplaceBits))),dtype = np.bool)
    coverMask = np.packbits(coverMaskBin)
    cover = np.bitwise_and(cover,coverMask)
    stego = np.bitwise_or(cover,payload)
    return stego

def complexity(img):
    alpha = []
    for i in range(0,8):
        bitPlaneMask = np.uint8(2**(i))
        plane = np.bitwise_and(img,bitPlaneMask)
        plane = np.uint8(plane/np.max(plane))
        a = arr_bpcs_complexity(plane)
        alpha.append(a)
        
    return alpha

def complexityConjugate(img,alphaThreshold,noOfReplaceBits):
    noOfImageBits = 8
    conjugateMap = np.zeros((noOfImageBits,),dtype=np.bool)
    for i in range(0,noOfReplaceBits):
        bitPlaneMask = np.uint8(2**(noOfImageBits-i-1))
        plane = np.bitwise_and(img,bitPlaneMask)
        if(plane.max()>1):
            plane = np.uint8(plane/plane.max())
        a = arr_bpcs_complexity(plane)
        if (a<alphaThreshold):
            conjugatePlane = conjugate(plane)*bitPlaneMask
            img = np.bitwise_and(img,np.bitwise_not(bitPlaneMask))
            img = np.bitwise_or(img,conjugatePlane)
            conjugateMap[i] = 1
    return img,np.packbits(conjugateMap)
    
def embedding(coverImage,payloadImage,n,x,y,displayImages = False):
    size = coverImage.shape
    alphaThreshold = 0.3
    rtemp = np.ones_like(coverImage)
    stegoImage = np.zeros_like(coverImage)
    for k in range(0,size[2]):
        for i in range(0,size[0],n):
            if(i+n<=size[0]):
                print(i,':',k)
                for j in range(0,size[1],n):
    #                print(j)
                    if(j+n<=size[1]):
                        alphaCover = complexity(coverImage[i:i+n,j:j+n,k])
                        noOfReplaceBits = x
                        for m in range(x+1,y+1):
                            if(alphaCover[m]>alphaThreshold):
                                noOfReplaceBits = m
                        payload, conjugateMap = complexityConjugate(payloadImage[i:i+n,j:j+n,k],alphaThreshold,noOfReplaceBits)
                        
                        stegoImage[i:i+n,j:j+n,k] = lsbEmbed(coverImage[i:i+n,j:j+n,k],payload,noOfReplaceBits)
                        stegoImage[i,j,k] = lsbEmbed(coverImage[i,j,k],conjugateMap,y)
                        rtemp[i:i+n,j:j+n,k] = noOfReplaceBits*rtemp[i:i+n,j:j+n,k]
                        
    if(displayImages):
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(2,3,1)
        ax.imshow(coverImage)
        ax.set_title('Cover Image')
        ax = plt.subplot(2,3,2)
        ax.imshow(payloadImage)
        ax.set_title('Payload Image')
        ax = plt.subplot(2,3,3)
        ax.imshow(stegoImage)
        ax.set_title('Stego-Image')
        plt.show();
    return stegoImage

def preprocessing(image,size = (1080,720)):
    image = cv2.resize(image,size)
    image = np.array(image,dtype = np.uint8)
    return image
    


n = 8
x = 3
y = 6

size = (1080,720)
#size = (80,80)
coverImage = preprocessing(cv2.imread('./images/img5.jpg'),size)
payloadImage = preprocessing(cv2.imread('./images/img3.png'),size)
stegoImage = embedding(coverImage,payloadImage,n,x,y,True)
cv2.imwrite('./images/coverImageBPCS53.png',coverImage)
cv2.imwrite('./images/stegoImageBPCS53.png',stegoImage)