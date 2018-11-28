# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 17:19:22 2018

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

c = '11'
p = '10'
coverImagePath = './input_images/image_'+c+'.png'
payloadImagePath = './input_images/image_'+p+'.png'
stegoImagePath = lambda method,c,p: './output_images/stego_'+method+'_'+c+'_'+p+'.png'
extractedCoverImagePath = lambda method,p: './output_images/cover_'+method+'_'+c+'.png'
extractedPayloadImagePath = lambda method,p: './output_images/payload_'+method+'_'+p+'.png'
plt.figure(figsize=(32, 18))
methods = {'lsb'}
ps = []
ms = []
for i in range(1,9):
    coverImage = cv2.imread(coverImagePath)
    payloadImage = cv2.imread(payloadImagePath)
    size = list(coverImage.shape)
    
    
    payloadImage = preprocessing(payloadImage,size)
    noOfReplaceBits = i
    stegoImage = lsbEmbed(coverImage,payloadImage,noOfReplaceBits)
    coverImage,payloadImage = lsbExtract(stegoImage,noOfReplaceBits)


    
    ax = plt.subplot(3,8,i)
    ax.imshow(stegoImage)
    ax.set_title('Stego-Image, n = '+str(i))
    ax = plt.subplot(3,8,i+8)
    ax.imshow(coverImage)
    ax.set_title('Cover Image, n = '+str(i))
    ax = plt.subplot(3,8,i+16)
    ax.imshow(payloadImage)
    ax.set_title('Payload Image, n = '+str(i))

    
    mse, snr, rmse,psnr = evaluate(coverImage,stegoImage)
    ps.append(psnr)
    ms.append(mse)
    print('MSE: ', mse)
    print('RMSE:', rmse)
    print('SNR: ', snr)
    print('PSNR:', psnr)
    cv2.imwrite(stegoImagePath(str(i),c,p),stegoImage)
    cv2.imwrite(extractedCoverImagePath(str(i),c),coverImage)
    cv2.imwrite(extractedPayloadImagePath(str(i),p),payloadImage)