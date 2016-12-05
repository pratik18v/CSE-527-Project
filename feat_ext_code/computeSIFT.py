#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 03:14:29 2016

@author: leena
"""

import cv2
import os
import numpy as np
#import matplotlib.pyplot as plt


######## load an image and compute dense SIFT #######


dir_name = "/home/leena/Documents/Projects/CV/Recipes/papers/preactivation_and_wide_resnet/res_net/res_net_code/"
os.chdir(dir_name)

#loading image data to know the size
data = np.load('X_train.npy')
len_data = len(data)

sift_descriptors_list = [] 

for x in range(len_data):
    
    dir_name = "/home/leena/Documents/Projects/CV/Recipes/papers/preactivation_and_wide_resnet/res_net/images/train"
    os.chdir(dir_name)

    file_name = "image%s" %x
    img = cv2.imread(file_name + ".jpg")
    gray= cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
    
    sift = cv2.xfeatures2d.SIFT_create()
    
    step_size = 1
    kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size) 
                                        for x in range(0, gray.shape[1], step_size)]
    
    #img=cv2.drawKeypoints(gray,kp, img)
    dense_feat = sift.compute(gray, kp)
    dense_desc = dense_feat[1]

    
    #desc_name = "desc%s" %x
    dense_desc = np.reshape(dense_desc, [128, 32, 32])
    sift_descriptors_list.append(dense_desc)

dir_name = "/home/leena/Documents/Projects/CV/Recipes/papers/preactivation_and_wide_resnet/res_net/sift_descriptors/train"
os.chdir(dir_name)  

#stacking all the descriptors  
sift_descriptors = np.vstack(sift_descriptors_list)
np.save("train_sift_descriptors.npy", sift_descriptors_list) 
    
#reading generated numpy array to check its dimension

sift_descriptors = np.load("train_sift_descriptors.npy")   
print("Length of generated descriptors is %s" %len(sift_descriptors)) 
#plt.figure(figsize=(20,10))
#plt.imshow(img)
#plt.show()    
#print("Done")