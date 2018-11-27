# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 01:21:07 2018

@author: Kiruthika
"""

import numpy as np
import cv2

def preprocessing(image,size = (1080,720)):
    image = cv2.resize(image,(size[1],size[0]))
    image = np.array(image,dtype = np.uint8)
    return image

def MSE(a,b):
    siz = a.shape
    mse = np.sum(np.square(a-b))/np.product(siz)
    return mse
def RMSE(a,b):
    rmse = (MSE(a,b))**0.5
    return rmse
def SNR(cover,stego):
    snr = 10*np.log10(MSE(cover,np.zeros_like(cover))/MSE(cover,stego))
    return snr
def PSNR(a,b):
    m = 255
    if(np.max(a)<=1):
        m = 1
    psnr = 10*np.log10(m**2/MSE(a,b))
    return psnr

def evaluate(coverImg,stegoImg):
    mse = MSE(coverImg,stegoImg)
    snr = SNR(coverImg,stegoImg)
    rmse = RMSE(coverImg,stegoImg)
    psnr = PSNR(coverImg,stegoImg)
    return mse, snr, rmse,psnr