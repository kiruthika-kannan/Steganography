
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import math

img_stego = cv2.imread('../output/Sparse_lena_stego.tif',cv2.IMREAD_GRAYSCALE)
rows , cols = img_stego.shape
img_stego_copy = img_stego.copy()
# print(type(img), img.shape)

# img_label = np.zeros((rows,cols), dtype = np.uint8)#array for labels
# print(img_label.shape)

# #print(img)


# secret_text = ['conference' ]
# asciival = []
# asciival = [ord(ch) for word in secret_text for ch in word]

# secret_arr = np.array(asciival)
# print('ascii value of secret data', secret_arr)

# print('length of secret data', len(secret_arr))
# s_len= len(secret_arr)



# In[2]:


def euclidean(pixel1, pixel2):
    '''calculate the euclidean distance
    input: numpy.arrays or lists
    return: euclidean distance
    '''
    dist = [(a - b)**2 for a, b in zip(pixel1, pixel2)]
    dist = math.sqrt(sum(dist))    
   
    return dist

#euclidean([0,0],[1,1])
 


# In[5]:


# prints intensities in range[0 to 255] with corresponding pixel count
def histogram(img):
    height = img.shape[0]
    width = img.shape[1]

    hist = np.zeros((256,),dtype='int')
    #print(hist.shape)   
    
    for i in np.arange(height):
        for j in np.arange(width):
            a = img.item(i,j)
            #print(a)
            hist[a] = hist[a]+1
    #print('Frequency count for Intensity range 0 to 255 :\n', hist)        
    return hist
    

t_hist = histogram(img_stego_copy)
#print(type(t))
#print(len(t))

One line compact if and else
L = [mapping-expression for element in source-list if filter-expression]

Example:
[i for i in range(10) if i%2==0]

# In[6]:


#intensity list of sparse pixels
thresh = 2
thresholded_freq_arr  = [val for val in t_hist if val < thresh ]
print('frequencies with pixel counts less than 2',thresholded_freq_arr)

intensity_thresh_list = list()
for i in range(0, len(t_hist)):
    val = t_hist[i]
    if val<thresh:
        intensity_thresh_list.append(i)   

intensity_thresh_arr = np.array(intensity_thresh_list)
print(type(intensity_thresh_arr))
print('intensities with pixel count less than 2',intensity_thresh_arr) 


# In[7]:


#Create empty secret array
s_len =10
z1 = np.empty(s_len , dtype = '<U10')        


# In[8]:


#LSB substitution for sparse pixels with large euclidean distance
def emd_extraction(image):
    t =0
    distance = 0
    for k in range(0,rows):
        for l in range(0,cols):
            for m in range(k,rows):
                for n in range(l,cols):
                    #print(image[m,n] in intensity_thresh_arr)
                    if (image[k,l] in intensity_thresh_arr) and (image[m,n] in intensity_thresh_arr):
                        distance = euclidean([k,l],[m,n])
                        print('in distance',distance)
                    if t == s_len:
                        print('in t == s_len')
                        break
                        #t = 0
                    #print(t,distance)
                    #lsb substitution
                    #if((distance>400) and (image[k,l] in intensity_thresh_arr) and (image[m,n] in intensity_thresh_arr) and (t<s_len)):
                    if((image[k,l] in intensity_thresh_arr) and (image[m,n] in intensity_thresh_arr) and (t<s_len)):
                        z2 = image[m,n]
                        z2_bin = format(z2, "08b")
                        z2_bin_msb = z2_bin[0:4]
                        z2_bin_lsb = z2_bin[4:8] 
                        z2_bin_split = z2_bin_msb + '0000'#reconstructed image pixel
                        z1[t] = z2_bin_lsb + '0000'#Reconstructed secret data
                        
                        image[m,n]=int(z2_bin_split, 2)
                        print([m,n])
                        print(z2,int(z2_bin_split, 2))
                        print(z2_bin_split,z1[t],t)
                        t=t+1
            if t == s_len:
                break  
    return image , z1

# 
        


# In[9]:


img_ex, z1_ex =  emd_extraction(img_stego_copy)
print(img_ex, z1_ex)


# In[19]:


print(z1_ex[1])
secret_out = np.zeros(s_len , np.int)
print(int(z1_ex[0],2))

for x in range(0, s_len):
    if z1_ex[x] not in '':
        #print('loop')
        secret_out[x] = int(z1_ex[x],2)
        
print(secret_out)


# In[15]:


#print(int('' , 2))

