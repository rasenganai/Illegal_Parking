
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
    img=cv2.resize(img,(224,224))
    for i in range(len(image[1][1])):
        data.append(transform(img,image[1][1][i]))
        target.append(int(image[1][-1][i]))
    return data,target

def preprocess(labels,path,batch_size):
    data,target=[],[]
    while True:
        np.random.shuffle(labels)
        for image in labels:
            d,t=image_trans(image,path)
            data+=d
            target+=t
            if len(data)>=batch_size:
                yield np.array(data),np.array(target)
                data=[]
                target=[]

#         yield np.array(data),np.array(target)


