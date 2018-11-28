# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 23:16:21 2018

@author: Kiruthika
"""

import cv2
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Evaluate stego-image')
parser.add_argument("cover", help='path to cover image')
parser.add_argument("stego", help='path to stego image')
args = parser.parse_args()
coverPath =args.cover
stegoPath = args.stego
coverImg = cv2.imread(coverPath)
stegoImg = cv2.imread(stegoPath)


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

    

mse = MSE(coverImg,stegoImg)
snr = SNR(coverImg,stegoImg)
rmse = RMSE(coverImg,stegoImg)
psnr = PSNR(coverImg,stegoImg)
print('MSE: ', mse)
print('RMSE:', rmse)
print('SNR: ', snr)
print('PSNR:', psnr)