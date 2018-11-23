#@Author-Ritu
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import math


# In[3]:


img = cv2.imread('../input/sun_crop.jpeg',cv2.IMREAD_GRAYSCALE)
#img = cv2.imread('../input/lena_re.jpg',cv2.IMREAD_GRAYSCALE)
rows , cols = img.shape
img_copy = img.copy()
print(type(img), img.shape)
print(img)
img_label = np.zeros((rows,cols), dtype = np.uint8)#array for labels
#print(img_label.shape)

#print(img)


secret_text = ['conference' ]
asciival = []
asciival = [ord(ch) for word in secret_text for ch in word]

secret_arr = np.array(asciival)
print('ascii value of secret data', secret_arr)

print('length of secret data', len(secret_arr))
s_len= len(secret_arr)


# In[4]:


#Decimal to binary

z1 = np.empty(s_len , dtype = '<U10')
#z1 =  format(secret_arr[0], "08b")
for x in range (0, s_len):
    z1[x] = format(secret_arr[x], '08b')
    
print(z1, type(z1[0]))    
    

def euclidean(pixel1, pixel2):
    '''calculate the euclidean distance
    input: numpy.arrays or lists
    return: euclidean distance
    '''
    dist = [(a - b)**2 for a, b in zip(pixel1, pixel2)]
    dist = math.sqrt(sum(dist))    
   
    return dist

#euclidean([0,0],[1,1])
 


# In[6]:


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
    
t_hist = histogram(img_copy)
#print(type(t))
print('length of t_hist',len(t_hist))


# In[9]:


#How to find sparese pixels?

thresh = 200#no of pixels
#thresh = 2
thresholded_freq_arr  = [val for val in t_hist if val < thresh ]
print('frequencies with pixel counts less than 300',thresholded_freq_arr)

#intensity list of sparse pixels
intensity_thresh_list = list()
for i in range(0, len(t_hist)):
    val = t_hist[i]
    if val<thresh:
        intensity_thresh_list.append(i)   

intensity_thresh_arr = np.array(intensity_thresh_list)
print(type(intensity_thresh_arr))
print('intensities with pixel count less than 200',intensity_thresh_arr) 


# In[16]:


# for i in range(0,rows):
#      for j in range(0,cols):
#         if img_copy[i,j] in thresholded_intensity:
#             img_label[i,j] = 1
            
#print(img_label)            


# In[11]:


# sparse pixels from Histogram
def embedding(image):
    t =0
    distance = 0
    for k in range(0,rows):
        for l in range(0,cols):
            for m in range(k,rows):
                for n in range(l,cols):
                    #print(image[m,n] in intensity_thresh_arr)
                    if (image[k,l] in intensity_thresh_arr) and (image[m,n] in intensity_thresh_arr):
                        distance = euclidean([k,l],[m,n])
                        print('in distance')
                    if t == s_len:
                        print('in t == s_len')
                        break
                        #t = 0
                    #print(t,distance)
                    #lsb substitution
                    if((distance>200) and (image[k,l] in intensity_thresh_arr) and (image[m,n] in intensity_thresh_arr) and (t<s_len)):
                        z2 = image[m,n]
                        z2_bin = format(z2, "08b")
                        z2_con = z2_bin[:4] + z1[t][0:4]
                        image[m,n]=int(z2_con, 2)
                        print([m,n])
                        print(z2,int(z2_con, 2))
                        print(z2_bin,z1[t],z2_con,t)
                        t=t+1
            if t == s_len:
                break  
    return image  
                                                       


# In[29]:


# def emd_embedding(image):
#     cv2.imwrite("../output/Sparse_stego.tif",img_copy)


# In[13]:


# print(2*n+1)
# print ((1-5)%7)
stego_image = embedding(img_copy)
print(stego_image)
cv2.imwrite("../output/sun_crop_stego.jpeg",stego_image)


# In[38]:


# img_stego = cv2.imread('../output/Sparse_stego.tif',cv2.IMREAD_GRAYSCALE)
# rows1 , cols1 = img_stego.shape
# smod_ex = 0

# def emd_extraction(image):
    
                
#                 print('extracted value', abc)
    
    

