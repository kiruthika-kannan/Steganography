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

c = '01'
p = '02'
coverImagePath = './input_images/image_'+c+'.png'
payloadImagePath = './input_images/image_'+p+'.png'
stegoImagePath = lambda method,c,p: './output_images/payload_'+method+'_'+c+'_'+p+'.png'
extractedCoverImagePath = lambda method,p: './output_images/cover_'+method+'_'+c+'.png'
extractedPayloadImagePath = lambda method,p: './output_images/payload_'+method+'_'+p+'.png'

method = 'lsb'
coverImage = cv2.imread(coverImagePath)
size = coverImage.shape
payloadImage = preprocessing(cv2.imread(payloadImagePath),size)
noOfReplaceBits = 3
stegoImage = lsbEmbed(coverImage,payloadImage,noOfReplaceBits,True)
coverImage,payloadImage = lsbExtract(stegoImage,noOfReplaceBits,True)

mse, snr, rmse,psnr = evaluate(coverImage,stegoImage)
print('MSE: ', mse)
print('RMSE:', rmse)
print('SNR: ', snr)
print('PSNR:', psnr)
cv2.imwrite(stegoImagePath(method,c,p),stegoImage)
cv2.imwrite(extractedCoverImagePath(method,c),coverImage)
cv2.imwrite(extractedPayloadImagePath(method,p),payloadImage)