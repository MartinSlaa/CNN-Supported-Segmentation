# -*- coding: utf-8 -*-
"""
Code has been created by Dr. Sreenivas Bhattiprolu https://github.com/bnsreenu/python_for_microscopists/blob/master/231_234_BraTa2020_Unet_segmentation/233_custom_datagen.py
Code has been adapted to my use. This is used to test a simple unet model.
"""
import os
from load_data import imageLoader
from unet_model import unet_model
import matplotlib.pyplot as plt
import glob
import random
import numpy as np
import segmentation_models_3D as sm
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import CSVLogger

# Import data
train_img_dir = "BraTS2020_TrainingData/input_data_128/train/images/"
train_mask_dir = "BraTS2020_TrainingData/input_data_128/train/masks/"

val_img_dir = "BraTS2020_TrainingData/input_data_128/val/images/"
val_mask_dir = "BraTS2020_TrainingData/input_data_128/val/masks/"

train_img_list=os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

val_img_list=os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

batch_size = 1

# crate image generator for training and validation. 
train_img_datagen = imageLoader(train_img_dir, 
                                train_img_list, 
                                train_mask_dir, 
                                train_mask_list, batch_size, )

val_img_datagen = imageLoader(val_img_dir, 
                              val_img_list, 
                              val_mask_dir, 
                              val_mask_list, batch_size)



# set step per epoch and val per epoch for training
steps_per_epoch = len(train_img_list)//batch_size
val_steps_per_epoch = len(val_img_list)//batch_size


# instantiate model with input shape. Print input shape and output shape 
model = unet_model(128, 128, 128, 3, 4)
print(model.input_shape)
print(model.output_shape)

# Compile model with optimizer Adam and Learning Rate of 0.0001.
model.compile(optimizer = tf.keras.optimizers.Adam(0.0001), loss='categorical_crossentropy',
              metrics= ['accuracy', sm.metrics.IOUScore(threshold=0.5)])


# Clear keras backend session

K.clear_session()

# create and write CSV log of training
csv_logger = CSVLogger('unet_3d_training.log', separator=',', append=False)

callbacks = [ tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.000001, verbose=1), csv_logger]

"""train model""" 
history=model.fit(train_img_datagen,
          steps_per_epoch=steps_per_epoch,
          epochs=200,
          verbose=1,
          validation_data=val_img_datagen,
          validation_steps=val_steps_per_epoch,
          callbacks= callbacks
          )

"""Save trained model"""
#model.save('models/unet_3d.hdf5')


# draw graph of training accuracy and training loss. 
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""Evaluate model"""

unet_3d = tensorflow.keras.models.load_model("models/unet_3d.hdf5",
                                  compile= False)

#Test mean IoU score for 8 images 
batch_size=8 
test_img_datagen = imageLoader(val_img_dir, val_img_list, 
                                val_mask_dir, val_mask_list, batch_size)

#
test_image_batch, test_mask_batch = test_img_datagen.__next__()

test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
test_pred_batch = unet_3d.predict(test_image_batch) # this should be run on GPU. Kernal crash when tried on CPU 
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

""" Check mean IoU score: results 0.75187665"""
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


"""Display two images with predicted segmentation"""

img_num1 = 82
img_num2 = 209

test_img1 = np.load("BraTS2020_TrainingData/input_data_128/val/images/image_"+str(img_num1)+".npy")

test_mask1 = np.load("BraTS2020_TrainingData/input_data_128/val/masks/mask_"+str(img_num1)+".npy")

test_mask_argmax1=np.argmax(test_mask1, axis=3)
test_img_input1 = np.expand_dims(test_img1, axis=0)
test_prediction1 = unet_3d.predict(test_img_input1)
test_prediction_argmax1=np.argmax(test_prediction1, axis=4)[0,:,:,:]

test_img2 = np.load("BraTS2020_TrainingData/input_data_128/val/images/image_"+str(img_num2)+".npy")

test_mask2 = np.load("BraTS2020_TrainingData/input_data_128/val/masks/mask_"+str(img_num2)+".npy")

test_mask_argmax2=np.argmax(test_mask2, axis=3)
test_img_input2 = np.expand_dims(test_img2, axis=0)
test_prediction2 = unet_3d.predict(test_img_input2)
test_prediction_argmax2=np.argmax(test_prediction2, axis=4)[0,:,:,:]


# Display images of original picture, ground truth and predicted segmentation
n_slice = 55
plt.figure(figsize=(20, 10))
plt.subplot(231)
plt.title('Original Image 82')
plt.imshow(test_img1[:,:,n_slice,0], cmap='gray')
plt.subplot(232)
plt.title('Ground truth 82')
plt.imshow(test_mask_argmax1[:,:,n_slice])
plt.subplot(233)
plt.title('Predicted Mask Image 82')
plt.imshow(test_prediction_argmax1[:,:, n_slice])
plt.subplot(234)
plt.title('Original Image 209')
plt.imshow(test_img2[:,:,n_slice,0], cmap='gray')
plt.subplot(235)
plt.title('Ground Truth 209')
plt.imshow(test_mask_argmax2[:,:,n_slice])
plt.subplot(236)
plt.title('Prediction Mask image 209')
plt.imshow(test_prediction_argmax2[:,:, n_slice])
plt.show()
