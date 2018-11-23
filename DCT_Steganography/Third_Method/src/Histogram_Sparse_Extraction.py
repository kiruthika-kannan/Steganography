
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import math

img_stego = cv2.imread('../output/sun_crop_stego.jpeg',cv2.IMREAD_GRAYSCALE)
rows , cols = img_stego.shape
img_stego_copy = img_stego.copy()




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
 


# In[3]:


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


# In[4]:


#intensity list of sparse pixels
thresh = 200
thresholded_freq_arr  = [val for val in t_hist if val < thresh ]
print('frequencies with pixel counts less than 200',thresholded_freq_arr)

intensity_thresh_list = list()
for i in range(0, len(t_hist)):
    val = t_hist[i]
    if val<thresh:
        intensity_thresh_list.append(i)   

intensity_thresh_arr = np.array(intensity_thresh_list)
print(type(intensity_thresh_arr))
print('intensities with pixel count less than 200',intensity_thresh_arr) 


# In[5]:


#Create empty secret array
s_len =10
z1 = np.empty(s_len , dtype = '<U10')        


# In[6]:


#sparse pixels from histogram 
def extraction(image):
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
                    if((distance>200) and (image[k,l] in intensity_thresh_arr) and (image[m,n] in intensity_thresh_arr) and (t<s_len)):
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


# In[7]:


img_ex, z1_ex = extraction(img_stego_copy)
print(img_ex, z1_ex)


# In[9]:



secret_out = np.zeros(s_len , np.int)
print(int(z1_ex[0],2))

for x in range(0, s_len):
    if z1_ex[x] not in '':
        secret_out[x] = int(z1_ex[x],2)
        
print('Ascii Value of extracted secret', secret_out)


# In[10]:


charval = []
charval = [chr(val) for val in secret_out ]
print(charval)

