
# coding: utf-8

# In[1]:


import numpy as np
import cv2

import matplotlib.pyplot as plt
from scipy.fftpack import  dct , idct

#%matplotlib qt
# from matplotlib.colors import Normalize
# import matplotlib.cm as cm


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


# In[3]:


#computing dct of gray cover
img_fl = np.float32(img2)/255.0  # float conversion/scale
#print(imf)
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

# In[5]:


img_sec_fl = np.float32(img_sec2)/255.0  # float conversion/scale
#print(imf)
#img_sec_dct = cv2.dct(img_sec_fl)           # the dct
img_sec_dct = dct(img_sec_fl , 2 , norm='ortho') 
print(img_sec_dct)


# In[6]:


weight = 0.5
img_dct_recreate = img_dct.copy()
#print(img_dct_recreate.shape)

img_dct_recreate[row-row1:row,col-col1:col] =   img_dct_recreate[row-row1:row,col-col1:col]  + weight *  img_sec_dct       
#img_dct_recreate[448:512,448:512] =   img_dct_recreate[448:512,448:512]  + weight *  img_sec_dct
print(img_dct_recreate.shape)


# In[7]:


#img_dct_inv = cv2.dct(img_dct_recreate,cv2.DCT_INVERSE)
img_dct_inv = idct(img_dct_recreate, 2 , norm='ortho')

print(img_dct_inv)


# In[8]:


img_inv_int =  np.uint8(img_dct_inv*255)
print(img_inv_int)


# In[9]:


cv2.imwrite('../output/Stego.jpg' , img_inv_int)


# # EXTRACT

# In[10]:


img_merge = cv2.imread('../output/Stego.jpg',cv2.IMREAD_GRAYSCALE)
plt.title('Stego image')
plt.imshow(img_merge)


# In[11]:


#Find DCT of stego image(combined with secret image)
img_merge_fl = np.float32(img_merge)/255.0  # float conversion/scale

img_merge_dct = dct(img_merge_fl , 2 , norm='ortho') 


# In[12]:


dct_diff = img_merge_dct - img_dct

dct_diff_div = dct_diff/weight

dct_diff_div_crop = dct_diff_div[row-row1:row,col-col1:col]
#dct_diff_div_crop = dct_diff_div[448:512,448:512]
print(dct_diff_div_crop.shape)


# In[13]:


img_ex_sec_inv = idct(dct_diff_div_crop, 2 , norm='ortho')
print(img_ex_sec_inv)


# In[14]:


img_ex_sec_inv_int = np.uint8(img_ex_sec_inv*255)
cv2.imwrite('../output/extract_secret.jpg' , img_ex_sec_inv_int)

