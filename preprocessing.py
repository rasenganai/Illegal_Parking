
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import pickle as pkl
import matplotlib.pyplot as plt


# In[55]:


def transform(img ,coor):
    im=img.copy()
    return cv2.rectangle(im,(int(coor[0]),int(coor[1])),(int(coor[2]),int(coor[3])),(0,0,0),cv2.FILLED)


# In[105]:


def image_trans(image,path):
    data=[]
    target=[]
    img=cv2.imread(path+image[0])/255.0
    for i in range(image[1][1].shape[0]):
        data.append(transform(img,image[1][1][i]))
        target.append(int(image[1][-1][i]))
    return data,target

def preprocess(labels,path,batch_size):
    data,target=[],[]
    
    for image in labels:
        d,t=image_trans(image,path)
        data+=d
        target+=t
        if len(data)>batch_size:
            yield np.array(data),np.array(target)
            data=[]
            target=[]
            
    return np.array(data),np.array(target)
    

