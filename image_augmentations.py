#!/usr/bin/env python
# coding: utf-8

# In[83]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle as pkl
import os 


# In[84]:


# labels=[]
# with open("./labels.pickle","rb") as file:
#     labels=pkl.load(file)


# In[85]:


#translation or cropping 1
def translation(img,coor):
    coor=np.array(coor)
    coor=coor.reshape((-1,4))
    img=cv2.resize(img,(224,224))
    rows,cols,_ = img.shape
    dst=img
    M = np.float32([[1,0,0],[0,1,50]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    M = np.float32([[1,0,0],[0,1,-50]])
    dst = cv2.warpAffine(dst,M,(cols,rows))
    dst=dst[50:-50,:,:]
    dst=cv2.resize(dst,(224,224))
    coor[:,1]=(coor[:,1]-50) * 224/ 124.0
    coor[:,3]=(coor[:,3]-50) * 224/ 124.0
    return dst,coor.reshape((-1,4))


# In[86]:


# Rotation and Scaling 2 
def rotationScale(img,coor,angle=5,scale=1.1):
    coor=np.array(coor)
    img=cv2.resize(img,(224,224))
    coor=coor.reshape((-1,2,2))
    rows,cols,_ = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,scale)
    dst = cv2.warpAffine(img,M,(cols,rows))
    dst=cv2.resize(dst,(224,224))
    coor=np.array(cv2.transform(coor,M))
#     print(coor)
    return dst,coor.reshape((-1,4))


# In[87]:


#FLIP 3
def flip(img,coor):
    coor=np.array(coor)
    img=cv2.resize(img,(224,224))
    coor=coor.reshape((-1,4))
    dst=cv2.flip(img,1)
    dst=cv2.resize(dst,(224,224))
#     print(coor)
    coor[:,0]=224-coor[:,0]
    coor[:,2]=224-coor[:,2]
    return dst,coor.reshape((-1,4))


# In[90]:


def augmentation(labels):
    augment_labels=[]
    count=1
    for image in labels:
        if image[1][0]==0:
            continue
        img=cv2.imread("./images/"+image[0])
        augs=[]
        augs.append(translation(img,image[1][1]))             #1
        augs.append(rotationScale(img,image[1][1]))           #2
        augs.append(rotationScale(img,image[1][1],angle=-7))  #3
        augs.append(rotationScale(img,image[1][1],angle=0))   #4
        augs.append(flip(img,image[1][1]))                    #5
        i=0
        for augim in augs:
            name=image[0].split(".")[0]+"_"+str(i+1)+".jpg"
            i+=1
            cv2.imwrite("./aug_images/"+name,augim[0])
            augment_labels.append((name,(image[1][0],augim[1],image[1][-1])))
#         print(count,end=" ")
        count+=1
    return augment_labels


# In[91]:


# augment_labels=augmentation(labels=labels)


# In[93]:


# len(augment_labels)+177


# In[94]:



# In[95]:


# a=np.random.randint(0,135)
# image=labels[a]
# img=cv2.imread("./Data/images/"+image[0])

# fig=plt.figur

# with open("./augments_label.pickle","wb") as file:
#     pkl.dump(augment_labels,file)e(figsize=(20,20))

# ax1=fig.add_subplot(1,4,2)
# im,coor=flip(img,image[1][1])
# for i in range(coor.shape[0]):
#     im=cv2.rectangle(im,(int(coor[i][0]),int(coor[i][1])),(int(coor[i][2]),int(coor[i][3])),(0,255,0),2)
# ax1.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))

# ax1=fig.add_subplot(1,4,3)
# im,coor=translation(img,image[1][1])
# for i in range(coor.shape[0]):
#     im=cv2.rectangle(im,(int(coor[i][0]),int(coor[i][1])),(int(coor[i][2]),int(coor[i][3])),(0,255,0),2)
# ax1.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))

# ax1=fig.add_subplot(1,4,4)
# im,coor=rotationScale(img,image[1][1],angle=7)
# for i in range(coor.shape[0]):
#     im=cv2.rectangle(im,(int(coor[i][0]),int(coor[i][1])),(int(coor[i][2]),int(coor[i][3])),(0,255,0),2)
# ax1.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))

# ax1=fig.add_subplot(1,4,1)
# coor=np.array(image[1][1])
# img=cv2.resize(img,(224,224))
# for i in range(coor.shape[0]):
#     img=cv2.rectangle(img,(int(coor[i][0]),int(coor[i][1])),(int(coor[i][2]),int(coor[i][3])),(0,255,0),2)

# ax1.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# # In[ ]:




