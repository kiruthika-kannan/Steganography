
# coding: utf-8

# In[21]:


import numpy as np
import math
import cv2

import matplotlib.pyplot as plt
from scipy.fftpack import  dct , idct

#%matplotlib qt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


# # MERGE

# In[2]:


img1 = cv2.imread('../input/Cameraman.png',cv2.IMREAD_GRAYSCALE)
row,col =img1.shape
print('size of original image', row,col)

img2 = img1.copy()


plt.title('Cover image')
plt.imshow(img1)

#cv2.imshow("R_Channel",img_r_c2) # For A Channel (Here's what You need)
#cv2.waitKey(0)


# computing dct of gray cover image

# In[3]:


#computing dct of gray cover
img_fl = np.float32(img2)/255.0  # float conversion/scale
#img_dct = cv2.dct(img_fl)           # the dct
img_dct = dct(img_fl , 2 , norm='ortho') 
print(img_dct)


# In[4]:


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

# In[6]:


img_sec_fl = np.float32(img_sec2)/255.0  # float conversion/scale
#print(imf)
#img_sec_dct = cv2.dct(img_sec_fl)           # the dct
img_sec_dct = dct(img_sec_fl , 2 , norm='ortho') 
#print(img_sec_dct)


# Assigning weight to embed secret image in cover image

# In[7]:


weight = 0.5
img_dct_recreate = img_dct.copy()
#print(img_dct_recreate.shape)

img_dct_recreate[row-row1:row,col-col1:col] =   img_dct_recreate[row-row1:row,col-col1:col]  + weight *  img_sec_dct       
print(img_dct_recreate.shape)


# In[8]:


#img_dct_inv = cv2.dct(img_dct_recreate,cv2.DCT_INVERSE)
img_dct_inv = idct(img_dct_recreate, 2 , norm='ortho')

#print(img_dct_inv)


# In[9]:


img_inv_int =  np.uint8(img_dct_inv*255)
print(img_inv_int)


# In[10]:


cv2.imwrite('../output/Stego_cameraman.jpg' , img_inv_int)


# # EXTRACT

# In[11]:


img_merge = cv2.imread('../output/Stego_cameraman.jpg',cv2.IMREAD_GRAYSCALE)
plt.title('Stego image')
plt.imshow(img_merge)


# In[12]:


#Find DCT of stego image(combined with secret image)
img_merge_fl = np.float32(img_merge)/255.0  # float conversion/scale

img_merge_dct = dct(img_merge_fl , 2 , norm='ortho') 


# In[13]:


dct_diff = img_merge_dct - img_dct

dct_diff_div = dct_diff/weight

dct_diff_div_crop = dct_diff_div[row-row1:row,col-col1:col]
#dct_diff_div_crop = dct_diff_div[448:512,448:512]
print(dct_diff_div_crop.shape)


# In[14]:


img_ex_sec_inv = idct(dct_diff_div_crop, 2 , norm='ortho')
print(img_ex_sec_inv)


# In[15]:


img_ex_sec_inv_int = np.uint8(img_ex_sec_inv*255)
cv2.imwrite('../output/extract_secret_cameraman.jpg' , img_ex_sec_inv_int)


# $$MSE = \frac{\Sigma \Sigma(In - Out)^2}{MN}$$
# 
# 
# M and N are the dimensions of the image

# In[16]:


mse =   ((img_sec1 - img_ex_sec_inv_int)**2).mean(axis=None)
print('Mean square error between input secret image and extracted secret image= ', mse)


# In[23]:


def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return(20 * math.log10(PIXEL_MAX / math.sqrt(mse)))

d=psnr(img_sec1,img_ex_sec_inv_int)
print('Peak signal-to-noise ratio is:' , d)

