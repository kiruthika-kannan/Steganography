
# coding: utf-8

# In[133]:


import numpy as np
import math
import cv2

import matplotlib.pyplot as plt
from scipy.fftpack import  dct , idct

#%matplotlib qt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


# # MERGE

# In[177]:


img1 = cv2.imread('../input/Room_512.jpeg',cv2.IMREAD_GRAYSCALE)
#img1 = cv2.imread('../input/House_225.jpeg',cv2.IMREAD_GRAYSCALE)
row,col =img1.shape
print('size of original image', row,col)

img2 = img1.copy()

plt.title('Cover image')
plt.imshow(img1)




#computing dct of gray cover
img_fl = np.float32(img2)/255.0  # float conversion/scale
#img_dct = cv2.dct(img_fl)           # the dct
img_dct = dct(img_fl , 2 , norm='ortho') 
print(img_dct)


# In[179]:


img_sec1 = cv2.imread('../input/lena_re.jpg',cv2.IMREAD_GRAYSCALE)
row1,col1 =img_sec1.shape
print('size of secret', row1,col1)

img_sec2 = img_sec1.copy()

plt.title('Cover image')
plt.imshow(img_sec1)


# Brief Steps for stego:
# embedSecretToJpeg(pixels, secret, fileout) {
#     blocks = splitBlocks(pixels);
#     coeffs = dct(blocks);
#     modified_coeffs = embedSecret(coeffs, secret);
#     saveCoefficients(modified_coeffs, fileout);
# }

# Computing DCT of secret image

# In[180]:


img_sec_fl = np.float32(img_sec2)/255.0  # float conversion/scale
#print(imf)
#img_sec_dct = cv2.dct(img_sec_fl)           # the dct
img_sec_dct = dct(img_sec_fl , 2 , norm='ortho') 
#print(img_sec_dct)


# Assigning weight to embed secret image in cover image


#weight = 0.007
weight = 0.5

img_dct_recreate = img_dct.copy()
#print(img_dct_recreate.shape)

img_dct_recreate[row-row1:row,col-col1:col] =   img_dct_recreate[row-row1:row,col-col1:col]  + weight *  img_sec_dct       
print(img_dct_recreate.shape)


# In[182]:


#img_dct_inv = cv2.dct(img_dct_recreate,cv2.DCT_INVERSE)
img_dct_inv = idct(img_dct_recreate, 2 , norm='ortho')

#print(img_dct_inv)



img_inv_int =  np.uint8(img_dct_inv*255)
print(img_inv_int)



#cv2.imwrite('../output/Stego_House_one.jpg' , img_inv_int)
cv2.imwrite('../output/Stego_Room_Two.jpg' , img_inv_int)


# # EXTRACT


#img_merge = cv2.imread('../output/Stego_House_one.jpg',cv2.IMREAD_GRAYSCALE)
img_merge = cv2.imread('../output/Stego_Room_Two.jpg',cv2.IMREAD_GRAYSCALE)
plt.title('Stego image')
plt.imshow(img_merge)


# In[186]:


#Find DCT of stego image(combined with secret image)
img_merge_fl = np.float32(img_merge)/255.0  # float conversion/scale

img_merge_dct = dct(img_merge_fl , 2 , norm='ortho') 


# In[187]:


dct_diff = img_merge_dct - img_dct

dct_diff_div = dct_diff/weight

dct_diff_div_crop = dct_diff_div[row-row1:row,col-col1:col]
#dct_diff_div_crop = dct_diff_div[448:512,448:512]
print(dct_diff_div_crop.shape)


# In[188]:


img_ex_sec_inv = idct(dct_diff_div_crop, 2 , norm='ortho')
print(img_ex_sec_inv)


# In[189]:


img_ex_sec_inv_int = np.uint8(img_ex_sec_inv*255)
cv2.imwrite('../output/extract_secret_Room_Two.jpg' , img_ex_sec_inv_int)
#cv2.imwrite('../output/extract_secret_House_One.jpg' , img_ex_sec_inv_int)





mse =   ((img_sec1 - img_ex_sec_inv_int)**2).mean(axis=None)
print('Mean square error between input secret image and extracted secret image= ', mse)


# In[191]:


def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return(20 * math.log10(PIXEL_MAX / math.sqrt(mse)))

d=psnr(img_sec1,img_ex_sec_inv_int)
print('Peak signal-to-noise ratio is:' , d)

