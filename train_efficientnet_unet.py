# -*- coding: utf-8 -*-
"""
Created on Tue May 24 16:18:28 2022

@author: marti
"""

from custom_metrics import dice_coef, dice_coef_necrotic, dice_coef_edema, dice_coef_enhancing, precision, sensitivity, specificity
from EfficientNet_model import conv_block, decoder_block, effienetnet_unet
from data_generator import DataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.callbacks import CSVLogger
import tensorflow as tf
import tensorflow.keras 
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
import os


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


#split data into training, testing and validation set
train_test_ids, val_ids = train_test_split(train_and_test_ids,test_size=0.2) 
train_ids, test_ids = train_test_split(train_test_ids,test_size=0.15) 


# Generate data
training_generator = DataGenerator(train_ids)
valid_generator = DataGenerator(val_ids)
test_generator = DataGenerator(test_ids)



# build and compile model
input_shape = (128,128,3)
effnet_unet = effienetnet_unet(input_shape)

effnet_unet.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), 
              dice_coef, precision, sensitivity, specificity, 
              dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing] )


# Clear backend session
K.clear_session()

# Logging training
csv_logger = CSVLogger('training_new_vgg16-unet.log', separator=',', append=False)

callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.000001, verbose=1),
                                  csv_logger]

# Train model
history =  effnet_unet.fit(training_generator,
                    epochs=100,
                    steps_per_epoch=len(train_ids),
                    callbacks= callbacks,
                    validation_data = valid_generator
                     )  

#Save model
#effnet_unet.save('trained_models/efficientb0_unet.hdf5')

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