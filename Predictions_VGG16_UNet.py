# Databricks notebook source

"""The original code for vgg16-unet implementation was written by Nikhil Tomar https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/TensorFlow/vgg16_unet.py
and the original code for datagenerator and code for displaying predicted images was written by Rastislav. at https://www.kaggle.com/code/rastislav/3d-mri-brain-tumor-segmentation-u-net/notebook
code has been adapted and modified for this project """

import os
import keras
import numpy as np
import nibabel as nib #pip install nilearn
import cv2 #pip install opencv-python
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', 
    2 : 'EDEMA',
    3 : 'ENHANCING' 

VOLUME_SLICES = 100 
VOLUME_START_AT = 22 


# if used on spyder, image path must be changed to "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/
TRAIN_DATASET_PATH = '/dbfs/mnt/mydata/BraTS/MICCAI_BraTS2020_TrainingData'

# if used on spyder, image path must be changed to "BraTS2020_TrainingData/MICCAI_BraTS2020_ValidationData/
VALIDATION_DATASET_PATH = '/dbfs/mnt/mydata/BraTS/MICCAI_BraTS2020_ValidationData'



train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]

def pathListIntoIds(dirList):
    x = []
    for i in range(0,len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/')+1:])
    return x

train_and_test_ids = pathListIntoIds(train_and_val_directories); 

    
train_test_ids, val_ids = train_test_split(train_and_test_ids,test_size=0.2) 
train_ids, test_ids = train_test_split(train_test_ids,test_size=0.15) 


# copy model from dbfs to databricks tmp folder used for databricks
#dbutils.fs.cp("dbfs:/mnt/mydata/BraTS/vgg16_unet.hdf5", "file:/tmp/vgg16_unet.hdf5")


# Load model
model = keras.models.load_model("trained_models/vgg16_unet.hdf5",
                                compile = False)



def predictByPath(case_path,case):
    files = next(os.walk(case_path))[2]
    X = np.empty((VOLUME_SLICES, 128, 128, 3))

    
    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_flair.nii');
    flair=nib.load(vol_path).get_fdata()
    
    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_t1ce.nii');
    ce=nib.load(vol_path).get_fdata() 
    
    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_t2.nii');
    t2=nib.load(vol_path).get_fdata() 


    
    for j in range(VOLUME_SLICES):
        X[j,:,:,0] = cv2.resize(flair[:,:,j+VOLUME_START_AT], (128,128))
        X[j,:,:,1] = cv2.resize(ce[:,:,j+VOLUME_START_AT], (128,128))
        X[j,:,:,2] = cv2.resize(t2[:,:,j+VOLUME_START_AT], (128,128))

    return model.predict(X/np.max(X), verbose=1)

# Load flair images as original image
def showPredictsById_flair(case, start_slice = 60):
    path = f"/dbfs/mnt/mydata/BraTS/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{case}"
    ground_truth = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii')).get_fdata()
    origImage = nib.load(os.path.join(path, f'BraTS20_Training_{case}_flair.nii')).get_fdata()
    p = predictByPath(path,case)

    core = p[:,:,:,1]
    edema= p[:,:,:,2]
    enhancing = p[:,:,:,3]

    plt.figure(figsize=(30, 70))
    f, axarr = plt.subplots(1,6, figsize = (18, 50)) 

    for i in range(6): 
        axarr[i].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (128, 128)), cmap="gray",                                       
                        interpolation='none')
    
    axarr[0].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (128, 128)), cmap="gray")
    axarr[0].title.set_text('Original Flair image')
    curr_ground_truth=cv2.resize(ground_truth[:,:,start_slice+VOLUME_START_AT], (128, 128), 
                                 interpolation = cv2.INTER_NEAREST)
    axarr[1].imshow(curr_ground_truth, cmap="Reds", interpolation='none', alpha=0.4) 
    axarr[1].title.set_text('Ground truth')
    axarr[2].imshow(p[start_slice,:,:,1:4], cmap="Reds", interpolation='none', alpha=0.4)
    axarr[2].title.set_text('prediction for all classes')
    axarr[3].imshow(core[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=0.4)
    axarr[3].title.set_text(f'{SEGMENT_CLASSES[1]} predicted')
    axarr[4].imshow(edema[start_slice,:,], cmap="OrRd", interpolation='none', alpha=0.4)
    axarr[4].title.set_text(f'{SEGMENT_CLASSES[2]} predicted')
    axarr[5].imshow(enhancing[start_slice,:,], cmap="OrRd", interpolation='none', alpha=0.4)
    axarr[5].title.set_text(f'{SEGMENT_CLASSES[3]} predicted')
    plt.show()



# Prediction on 6 random images and displays the results
showPredictsById_flair(case=test_ids[2][-3:])
showPredictsById_flair(case=test_ids[30][-3:])
showPredictsById_flair(case=test_ids[22][-3:])
showPredictsById_flair(case=test_ids[18][-3:])
showPredictsById_flair(case=test_ids[19][-3:])
showPredictsById_flair(case=test_ids[37][-3:])



# Load T1ce image as original image
def showPredictsById_t1ce(case, start_slice = 60):
    path = f"/dbfs/mnt/mydata/BraTS/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{case}"
    ground_truth = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii')).get_fdata()
    origImage = nib.load(os.path.join(path, f'BraTS20_Training_{case}_t1ce.nii')).get_fdata()
    p = predictByPath(path,case)

    core = p[:,:,:,1]
    edema= p[:,:,:,2]
    enhancing = p[:,:,:,3]

    plt.figure(figsize=(25, 70))
    f, axarr = plt.subplots(1,6, figsize = (18, 50)) 

    for i in range(6): # for each image, add brain background
        axarr[i].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (128, 128)), cmap="gray",
                        interpolation='none')
    
    axarr[0].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (128, 128)), cmap="gray")
    axarr[0].title.set_text('Original T1ce image')
    curr_ground_truth=cv2.resize(ground_truth[:,:,start_slice+VOLUME_START_AT], (128, 128),
                                 interpolation = cv2.INTER_NEAREST)
    axarr[1].imshow(curr_ground_truth, cmap="Reds", interpolation='none', alpha=0.4) # ,alpha=0.3,cmap='Reds'
    axarr[1].title.set_text('Ground truth')
    axarr[2].imshow(p[start_slice,:,:,1:4], cmap="Reds", interpolation='none', alpha=0.4)
    axarr[2].title.set_text('Prediction for all classes')
    axarr[3].imshow(core[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=0.4)
    axarr[3].title.set_text(f'{SEGMENT_CLASSES[1]} predicted')
    axarr[4].imshow(edema[start_slice,:,], cmap="OrRd", interpolation='none', alpha=0.4)
    axarr[4].title.set_text(f'{SEGMENT_CLASSES[2]} predicted')
    axarr[5].imshow(enhancing[start_slice,:,], cmap="OrRd", interpolation='none', alpha=0.4)
    axarr[5].title.set_text(f'{SEGMENT_CLASSES[3]} predicted')
    plt.show()



# Model Prediction on 6 random images
showPredictsById_t1ce(case=test_ids[2][-3:])
showPredictsById_t1ce(case=test_ids[30][-3:])
showPredictsById_t1ce(case=test_ids[22][-3:])
showPredictsById_t1ce(case=test_ids[18][-3:])
showPredictsById_t1ce(case=test_ids[19][-3:])
showPredictsById_t1ce(case=test_ids[37][-3:])



# Load T2 image as original image
def showPredictsById_t2(case, start_slice = 60):
    path = f"/dbfs/mnt/mydata/BraTS/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{case}"
    ground_truth = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii')).get_fdata()
    origImage = nib.load(os.path.join(path, f'BraTS20_Training_{case}_t2.nii')).get_fdata()
    p = predictByPath(path,case)

    core = p[:,:,:,1]
    edema = p[:,:,:,2]
    enhancing = p[:,:,:,3]

    plt.figure(figsize=(30, 70))
    f, axarr = plt.subplots(1,6, figsize = (18, 50)) 

    for i in range(6): 
        axarr[i].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (128, 128)),cmap="gray", 
                        interpolation='none')
    
    axarr[0].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (128, 128)),cmap="gray")
    axarr[0].title.set_text('Original T2 image')
    curr_ground_truth=cv2.resize(ground_truth[:,:,start_slice+VOLUME_START_AT], (128, 128),
                                 interpolation = cv2.INTER_NEAREST)
    axarr[1].imshow(curr_ground_truth, cmap="Reds", interpolation='none', alpha=0.4) # 
    axarr[1].title.set_text('Ground truth')
    axarr[2].imshow(p[start_slice,:,:,1:4], cmap="Reds", interpolation='none', alpha=0.4)
    axarr[2].title.set_text('Prediction all classes')
    axarr[3].imshow(core[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=0.4)
    axarr[3].title.set_text(f'{SEGMENT_CLASSES[1]} predicted')
    axarr[4].imshow(edema[start_slice,:,], cmap="OrRd", interpolation='none', alpha=0.4)
    axarr[4].title.set_text(f'{SEGMENT_CLASSES[2]} predicted')
    axarr[5].imshow(enhancing[start_slice,:,], cmap="OrRd", interpolation='none', alpha=0.4)
    axarr[5].title.set_text(f'{SEGMENT_CLASSES[3]} predicted')
    plt.show()



# Model Prediction on 6 random images
showPredictsById_t2(case=test_ids[2][-3:])
showPredictsById_t2(case=test_ids[30][-3:])
showPredictsById_t2(case=test_ids[22][-3:])
showPredictsById_t2(case=test_ids[18][-3:])
showPredictsById_t2(case=test_ids[19][-3:])
showPredictsById_t2(case=test_ids[37][-3:])
