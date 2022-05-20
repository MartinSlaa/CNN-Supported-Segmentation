# -*- coding: utf-8 -*-
"""

Code has been created by Dr. Sreenivas Bhattiprolu https://github.com/bnsreenu/python_for_microscopists/blob/master/231_234_BraTa2020_Unet_segmentation/233_custom_datagen.py
Code has been adapted to my use. This is used to test a simple unet model.

"""

import os
import random
import numpy as np 
import matplotlib.pyplot as plt




# import the data into training and testing directory
train_img_dir = "BraTS2020_TrainingData/input_data_128/train/images/"
train_mask_dir = "BraTS2020_TrainingData/input_data_128/train/masks/"

val_img_dir = "BraTS2020_TrainingData/input_data_128/val/images/"
val_mask_dir = "BraTS2020_TrainingData/input_data_128/val/masks/"

train_img_list=os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

val_img_list=os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

# print lenght of each list: 275, 275, 69, 69 respectively 
print(len(train_img_list))
print(len(train_mask_list))
print(len(val_img_list))
print(len(val_mask_list))

# function to load image from numpy array. 
def load_img(img_dir, img_list):
    images=[]
    for i, image_name in enumerate(img_list):    
        if (image_name.split('.')[1] == 'npy'):
            
            image = np.load(img_dir+image_name)
                      
            images.append(image)
    images = np.array(images)
    
    return(images)



# load images and set images and mask in to X and Y. 
def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):

    L = len(img_list)

    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
                       
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

            yield (X,Y) # use yield to create a generator_object which can be iterated through.      

            batch_start += batch_size   
            batch_end += batch_size


# create training and testing image generator with a batch size of 1.
batch_size = 1

train_img_datagen = imageLoader(train_img_dir, 
                                train_img_list, 
                                train_mask_dir, 
                                train_mask_list, batch_size, )

val_img_datagen = imageLoader(val_img_dir, 
                              val_img_list, 
                              val_mask_dir, 
                              val_mask_list, batch_size)





steps_per_epoch = len(train_img_list)//batch_size
val_steps_per_epoch = len(val_img_list)//batch_size

# Test image generator and display one image of each modality and mask. 
img, msk = train_img_datagen.__next__()


img_num = random.randint(0,img.shape[0]-1)
test_img=img[img_num]
test_mask=msk[img_num]


n_slice= 55
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