# -*- coding: utf-8 -*-
"""
The original code for vgg16-unet implementation was written by Nikhil Tomar https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/TensorFlow/resnet50_unet.py
and the original code for datagenerator was written by original code written by Rastislav. at https://www.kaggle.com/code/rastislav/3d-mri-brain-tumor-segmentation-u-net/notebook

Code has been modified for this project. 
"""

from custom_metrics import dice_coef, dice_coef_necrotic, dice_coef_edema, dice_coef_enhancing, precision, sensitivity, specificity
from resnet50_unet_model import conv_block, decoder_block, resnet_unet
from data_generator import DataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard
from tensorflow.keras.callbacks import CSVLogger
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf


# set data path
TRAIN_DATASET_PATH = 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'

VALIDATION_DATASET_PATH = 'BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'

# Set image size
IMG_SIZE=128

# load data into directory
train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]

def pathListIntoIds(dirList):
    x = []
    for i in range(0,len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/')+1:])
    return x

train_and_test_ids = pathListIntoIds(train_and_val_directories); 

# Split data into train, test and validation set    
train_test_ids, val_ids = train_test_split(train_and_test_ids,test_size=0.2) 
train_ids, test_ids = train_test_split(train_test_ids,test_size=0.15) 


#generate data
training_generator = DataGenerator(train_ids)
valid_generator = DataGenerator(val_ids)
test_generator = DataGenerator(test_ids)


# build and compile model
input_shape = (128,128,3)
resnet50_unet = resnet_unet(input_shape)

resnet50_unet.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), 
              dice_coef, precision, sensitivity, specificity, 
              dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing] )


# Clear keras backend session
K.clear_session()

# logg training
csv_logger = CSVLogger('training_resnet-unet.log', separator=',', append=False)


callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.000001, verbose=1),
                                  csv_logger]

# Start training
history =  resnet50_unet.fit(training_generator,
                    epochs=100,
                    steps_per_epoch=len(train_ids),
                    callbacks= callbacks,
                    validation_data = valid_generator
                     )  

# save model
#model.save('trained_models/resnet50_unet.hdf5')

# draw graph of training and validation accuracy and loss
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