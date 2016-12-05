#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:59:04 2016

@author: leena
"""

import numpy as np
import cv2
import os

######## read data from Numpy array and convert to image #######

data = np.load('X_test.npy')
len_data = len(data)
#labels = np.load('y_test.npy')

dir_name = "/home/leena/Documents/Projects/CV/Recipes/papers/preactivation_and_wide_resnet/images"
os.chdir(dir_name)

for x in range(len_data):
    
    img_data = data[x]

    ####### Merging the r,g,b channels #######  
    rchannel = img_data[2,:,:]
    gchannel = img_data[1,:,:]
    bchannel = img_data[0,:,:]
    file_name = "image%s" %x
    img = cv2.merge([rchannel, gchannel, bchannel])  
    
    cv2.imwrite(file_name + ".jpg", img)

    
####### To see the image #######
#cv2.imshow("Display Image", uintdata)
#cv2.waitKey(100000)
#cv2.destroyAllWindows()
