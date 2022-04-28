# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:11:29 2022

@author: marti
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import random

def load_img(img_dir, img_list):
    images = []
    for i, image_name in enumerate(img_list):
        if (image_name.split('.')[1] == 'npy'):
            
            image = np.load(img_dir+image_name)
            
            images.append(image)
    
    images = np.array(images)
    
    return(images)

def image_loader(img_dir, img_list, mask_dir, mask_list, batch_size):
    L = len(img_list)
    
    while True:
        batch_start = 0
        batch_end = batch_size
        
        while batch_start < L:
            limit = min(batch_end, L)
            
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])
            
            yield (X,Y)
            
            batch_start += batch_size
            batch_end += batch_size
            
def image_loader_val(img_dir, img_list, batch_size):
    L = len(img_list)
    
    while True:
        batch_start = 0
        batch_end = batch_size
        
        while batch_start < L:
            limit = min(batch_end, L)
            
            X = load_img(img_dir, img_list[batch_start:limit])
           
            
            yield (X)
            
            batch_start += batch_size
            batch_end += batch_size
            
train_img_dir = 'Data_3channels_train/images/'
train_mask_dir = 'Data_3channels_train/masks/'
val_img_dir = 'Data_3channels_val/images/'

train_img_list=os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

batch_size = 2

train_img_datagen = image_loader(train_img_dir, train_img_list, 
                                train_mask_dir, train_mask_list, batch_size)

# Verify image generator
img, msk = train_img_datagen.__next__()

val_img_list = os.listdir(val_img_dir)
val_img_datagen = image_loader_val(val_img_dir, val_img_list, batch_size)

# display images

#img_num = random.randint(0,img.shape[0]-1)
test_img=img[1]
test_mask=msk[1]
test_mask=np.argmax(test_mask, axis=3)

n_slice=random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(test_img[:,:,n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(test_img[:,:,n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(test_img[:,:,n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')
plt.show()
