# -*- coding: utf-8 -*-
"""
The original code for EffcientNetB0-unet implementation was written by Nikhil Tomar https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/TensorFlow/resnet50_unet.py
Code has been modified for this project.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, BatchNormalization, Activation, Input, Concatenate
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.utils import plot_model



def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(inputs, skip, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters)
    return x

def effienetnet_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained Encoder """
    encoder = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)

    for layer in encoder.layers[:15]:
          layer.trainable = False
    
    s1 = encoder.get_layer("input_7").output                     
    s2 = encoder.get_layer("block2a_expand_activation").output    
    s3 = encoder.get_layer("block3a_expand_activation").output    
    s4 = encoder.get_layer("block4a_expand_activation").output    

    """ Bottleneck """
    b1 = encoder.get_layer("block6a_expand_activation").output    

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                               
    d2 = decoder_block(d1, s3, 256)                               
    d3 = decoder_block(d2, s2, 128)                               
    d4 = decoder_block(d3, s1, 64)                                

    """ Output """
    outputs = Conv2D(4, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="EfficientNetB0_UNET")
    return model

input_shape = (128,128, 3)
efficientNet_UNet = effienetnet_unet(input_shape)
efficientNet_UNet.summary()
print(efficientNet_UNet.input_shape)
print(efficientNet_UNet.output_shape)


plot_model(efficientNet_UNet, 
           show_shapes = True,
           show_layer_names = True, 
           rankdir = 'TB', 
           expand_nested = False, 
           dpi = 70)