# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 01:48:25 2018

@author: Kiruthika
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 10:37:13 2018

@author: Kiruthika
"""
import numpy as np
from matplotlib import pyplot as plt
import cv2
def lsbEmbed(cover,payload,noOfReplaceBits):
    noOfImageBits = 8
    payload = np.right_shift(payload,(noOfImageBits-noOfReplaceBits))
    coverMaskBin = np.array(np.concatenate((np.ones(noOfImageBits-noOfReplaceBits),np.zeros(noOfReplaceBits))),dtype = np.bool)
    coverMask = np.packbits(coverMaskBin)
    cover = np.bitwise_and(cover,coverMask)
    stego = np.bitwise_or(cover,payload)
    return stego
    
    
def embedding(coverImage,payloadImage,n,x,y,displayImages = False):
    size = coverImage.shape
    stegoImage = np.zeros_like(coverImage)
    edgeImage = cv2.Canny(coverImage,100,200) #get edge image
    plt.imshow(edgeImage)
    plt.show()
    for k in range(0,size[2]):
        for i in range(0,size[0]):
#            print(i,'----------------------')
            for j in range(0,size[1],n):
#                print(j)
                if(j+n<=size[1]):
                    status = 0
                    for m in range(1,n):
                        if(edgeImage[i,j+m] == 0):
                            stegoImage[i,j+m,k] = lsbEmbed(coverImage[i,j+m,k],payloadImage[i,np.int64(j*(n-1)/n+m-1),k],x)
                        else:
                            status = status + 2**(m-1)
                            stegoImage[i,j+m,k] = lsbEmbed(coverImage[i,j+m,k],payloadImage[i,np.int64(j*(n-1)/n+m-1),k],y)
                    stegoImage[i,j,k] = lsbEmbed(coverImage[i,j,k],status,n-1)
                        
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
    


n = 3
x = 2
y = 4
size = (1080,720)
coverImage = preprocessing(cv2.imread('./images/img5.jpg'),size)
payloadImage = preprocessing(cv2.imread('./images/img3.png'),(np.int64(size[0]*(n-1)/n),size[1]))
stegoImage = embedding(coverImage,payloadImage,n,x,y,True)
cv2.imwrite('./images/coverImageEdge53.png',coverImage)
cv2.imwrite('./images/stegoImageEdge53.png',stegoImage)