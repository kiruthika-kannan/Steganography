# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:16:08 2018

@author: Kiruthika
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 01:18:49 2018

@author: Kiruthika
"""
import numpy as np
from matplotlib import pyplot as plt
import cv2
from utils import preprocessing, evaluate

from least_significant_bit_embedding import embedding as lsbEmbed
from least_significant_bit_extraction import extraction as lsbExtract
from pvd_embedding import embedding as pvdEmbed
from pvd_extraction import extraction as pvdExtract
from edge_embedding import embedding as edgeEmbed
from edge_extraction import extraction as edgeExtract
from bpcs_embedding import embedding as bpcsEmbed
from bpcs_extraction import extraction as bpcsExtract

import argparse
parser = argparse.ArgumentParser(description='LSB,PVD,Edge,BPCS methods. stego and extracted output in output_images_test folder')
parser.add_argument("cover", help='path to cover image')
parser.add_argument("stego", help='path to stego image')
args = parser.parse_args()
coverImagePath =args.cover
payloadImagePath = args.stego

c = '01'
p = '02'
stegoImagePath = lambda method,c,p: './output_images_test/stego_'+method+'.png'
extractedCoverImagePath = lambda method,p: './output_images_test/cover_'+method+'.png'
extractedPayloadImagePath = lambda method,p: './output_images_test/payload_'+method+'.png'

methods = {'pvd','lsb','bpcs','edge'}
for method in methods:
    print('Method: ', method)
    coverImage = cv2.imread(coverImagePath)
    payloadImage = cv2.imread(payloadImagePath)
    size = list(coverImage.shape)
    
    if method == 'lsb':
        payloadImage = preprocessing(payloadImage,size)
        noOfReplaceBits = 3
        stegoImage = lsbEmbed(coverImage,payloadImage,noOfReplaceBits)
        coverImage,payloadImage = lsbExtract(stegoImage,noOfReplaceBits)
        
    elif method == 'pvd':
        size[1] = np.int(size[1]/2)
        payloadImage = preprocessing(payloadImage,size)
        stegoImage = pvdEmbed(coverImage,payloadImage)
        coverImage,payloadImage = pvdExtract(stegoImage)
        
    elif method == 'edge':
        n = 3
        x = 3
        y = 4
        size[1] = np.int64(size[1]*(n-1)/n)
        payloadImage = preprocessing(payloadImage,size)
        stegoImage = edgeEmbed(coverImage,payloadImage,n,x,y)
        coverImage,payloadImage = edgeExtract(stegoImage,n,x,y)
        
    elif method == 'bpcs':
        n = 8
        x = 3
        y = 4
        payloadImage = preprocessing(payloadImage,size)
        stegoImage = bpcsEmbed(coverImage,payloadImage,n,x,y)
        coverImage,payloadImage = bpcsExtract(stegoImage,n,x,y)
    
    mse, snr, rmse,psnr = evaluate(coverImage,stegoImage)
    print('MSE: ', mse)
    print('RMSE:', rmse)
    print('SNR: ', snr)
    print('PSNR:', psnr)
    cv2.imwrite(stegoImagePath(method,c,p),stegoImage)
    cv2.imwrite(extractedCoverImagePath(method,c),coverImage)
    cv2.imwrite(extractedPayloadImagePath(method,p),payloadImage)