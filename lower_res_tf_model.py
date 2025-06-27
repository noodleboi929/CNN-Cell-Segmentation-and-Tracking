import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tifffile import imread, imwrite
import cv2
import math
import os
from pathlib import Path
import sys

# We will divide the original image into 4 parts to lower resolution and lower memory strain

# I will define the model using a Functional API

inputs = keras.Input(shape = (256,256, 1))

# Input tensor: (256, 256, 1)
# Output tensor: (254, 254, 64)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 64 

conv_1 = layers.Conv2D(64, (3,3), (1,1), activation= 'relu', use_bias = True)(inputs)

print("conv_1 completed")

print(conv_1.shape)

# Input tensor: (254, 254, 64)
# Output tensor: (252, 252, 64)
# Kernel Dimensions: 3 X 3 X 64
# Number of Kernels: 64

conv_2 = layers.Conv2D(64, (3,3), (1,1), activation = 'relu', use_bias = True)(conv_1)

#skip_connection_1 = conv_2.output

print("conv_2 completed")

print(conv_2.shape)

# Input tensor: (252, 252, 64)
# Output tensor: (250, 250, 64)
# Kernel Dimensions: 3 X 3 X 64
# Number of Kernels: 64

conv_3 = layers.Conv2D(64, (3,3), (1,1), activation = 'relu', use_bias = True)(conv_2)

print("conv_3 completed")

print(conv_3.shape)

# Input tensor: (250, 250, 64)
# Output tensor: (124.5 => 124, 124.5 => 124, 64)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 1
# Stride: 2 

conv_pool_1 = layers.AveragePooling2D((3,3), (2,2))(conv_3)

print("conv_pool_1 completed")

print(conv_pool_1.shape)

# Input tensor: (124.5 => 124, 124.5 => 124, 64)
# Output tensor: (122, 122, 128)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 128

conv_4 = layers.Conv2D(128, (3,3), (1,1), activation = 'relu', use_bias = True)(conv_pool_1)

print("conv_4 completed")

print(conv_4.shape)

# Input tensor: (122, 122, 128)
# Output tensor: (120, 120, 128)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 128

conv_5 = layers.Conv2D(128, (3,3), (1,1), activation = 'relu', use_bias = True)(conv_4)

# skip_connection_2 = conv_4.output

print("conv_5 completed")

print(conv_5.shape)

# Input tensor: (120, 120, 128)
# Output tensor: (118, 118, 128)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 128

conv_6 = layers.Conv2D(128, (3,3), (1,1), activation = 'relu', use_bias = True)(conv_5)

print("conv_6 completed")

print(conv_6.shape)

# Input tensor: (118, 118, 128)
# Output tensor: (58.5 => 58, 58.5 => 58, 128)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 1
# Stride: 2

conv_pool_layer_2 = layers.AveragePooling2D((3,3), (2,2))(conv_6)

print("conv_pool_layer_2 completed")

print(conv_pool_layer_2.shape)

# Input tensor: (58.5 => 58, 58.5 => 58, 128)
# Output tensor: (56, 56, 256)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 256

conv_7 = layers.Conv2D(256, (3,3), (1,1), activation = 'relu', use_bias = True)(conv_pool_layer_2)

print("conv_7 completed")

print(conv_7.shape)

# Input tensor: (56, 56, 256)
# Output tensor: (54, 54, 256)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 256

conv_8 = layers.Conv2D(256, (3,3), (1,1), activation = 'relu', use_bias = True)(conv_7)

print("conv_8 completed")

print(conv_8.shape)

# Input tensor: (54, 54, 256)
# Output tensor: (52, 52, 256)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 256

conv_9 = layers.Conv2D(256, (3,3), (1,1), activation = 'relu', use_bias = True)(conv_8)

print("conv_9 completed")

print(conv_9.shape)

# Input tensor: (52, 52, 256)
# Output tensor: (104, 104, 256)
# Kernel Dimensions: 2 X 2
# Number of Kernels: 256
# stride: 2

up_conv_1 = layers.Conv2DTranspose(256, (2,2), (2,2), activation = 'relu', use_bias = True)(conv_9)

print("up_conv_1 completed")

print(up_conv_1.shape)

# Input tensor: (104, 104, 256) & (118, 118, 128)
# Output tensor: (104, 104, 384)

resize_4 = layers.Resizing(104,104)(conv_6)

concat_1 = layers.Concatenate(axis = -1)([up_conv_1, resize_4])

print("concat_1 completed")

print(concat_1.shape)

# Input tensor: (104, 104, 384)
# Output tensor: (98, 98, 256)
# Kernel Dimensions: 7 X 7
# Number of Kernels: 256

conv_10 = layers.Conv2D(256, (7,7), (1,1), activation= 'relu', use_bias = True)(concat_1)

print("conv_10 completed")

print(conv_10.shape)

# Input tensor: (98, 98, 256)
# Output tensor: (92, 92, 256)
# Kernel Dimensions: 7 X 7
# Number of Kernels: 256

conv_11 = layers.Conv2D(256, (7,7), (1,1), activation = 'relu', use_bias = True)(conv_10)

print("Conv_11 completed")

print(conv_11.shape)

# Input tensor: (92, 92, 256) & (120, 120, 128)
# Output tensor: (92, 92, 384)


resize_11_12 = layers.Resizing(92,92)(conv_5)

concat_11_12 = layers.Concatenate(axis = -1)([conv_11, resize_11_12])

print("concat_11_12 completed")

print(concat_11_12.shape)

# Input tensor: (92, 92, 384)
# Output tensor: (86, 86, 256)
# Kernel Dimensions: 7 X 7
# Number of Kernels: 256

conv_12 = layers.Conv2D(256, (7,7), (1,1), activation = 'relu', use_bias = True)(concat_11_12)

print("Conv_12 completed")

print(conv_12.shape)

# Input tensor: (86, 86, 256)
# Output tensor: (172, 172, 256)
# Kernel Dimensions: 2 X 2
# Number of Kernels: 256
# Stride: 2

up_conv_2 = layers.Conv2DTranspose(256, (2,2), (2,2), activation = 'relu', use_bias = True)(conv_12)

print("up_conv_2 completed")

print(up_conv_2.shape)

# Input tensor: (172, 172, 256) & (250, 250, 64)
# Output tensor: (172, 172, 320)

resize_3 = layers.Resizing(172,172)(conv_3)

concat_2 = layers.Concatenate(axis = -1)([up_conv_2, resize_3])

print("concat_2 completed")

print(concat_2.shape)

# Input tensor: (172, 172, 320)
# Output tensor: (166, 166, 128)
# Kernel Dimensions: 7 X 7
# Number of Kernels: 512

conv_13 = layers.Conv2D(128, (7,7), (1,1), activation = 'relu', use_bias = True)(concat_2)

print("conv_13 completed")

print(conv_13.shape)

# Input tensor: (166, 166, 128)
# Output tensor: (160, 160, 128)
# Kernel Dimensions: 7 X 7
# Number of Kernels: 512

conv_14 = layers.Conv2D(128, (7,7), (1,1), activation = 'relu', use_bias = True)(conv_13)

print("conv_14 completed")

print(conv_14.shape)

# Input tensor: (160, 160, 128) & (252, 252, 64)
# Output tensor: (160, 160, 192)

resize_14_15 = layers.Resizing(160,160)(conv_2)

concat_14_15 = layers.Concatenate(axis = -1)([conv_14, resize_14_15])

print("concat_14_15 completed")

print(concat_14_15.shape)

# Input tensor: (160, 160, 192)
# Output tensor: (154, 154, 128)
# Kernel Dimensions: 7 X 7
# Number of Kernels: 512

conv_15 = layers.Conv2D(128, (7,7), (1,1), activation = 'relu', use_bias = True)(concat_14_15)

print("conv_15 completed")

print(conv_15.shape)

# Input tensor: (154, 154, 128)
# Output tensor: (76.5 => 76, 76.5 => 76, 128)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 1
# Stride: 2

conv_pool_layer_3 = layers.AveragePooling2D((3,3), (2,2))(conv_15)

print("conv_pool_layer_3 completed")

print(conv_pool_layer_3.shape)

# Input tensor: (76.5 => 76, 76.5 => 76, 128)
# Output tensor: (70, 70, 128)
# Kernel Dimensions: 7 X 7
# Number of Kernels: 128

conv_16 = layers.Conv2D(128, (7,7), (1,1), activation = 'relu', use_bias = True)(conv_pool_layer_3)

print("conv_16 completed")

print(conv_16.shape)

# Input tensor: (70, 70, 128)
# Output tensor: (65, 65, 64)
# Kernel Dimensions: 6 X 6
# Number of Kernels: 64

conv_17 = layers.Conv2D(64, (6,6), (1,1), activation = 'relu', use_bias = True)(conv_16)

print("conv_17 completed")

print(conv_17.shape)


# Input tensor: (65, 65, 64)
# Output tensor: (130, 130, 64)
# Kernel Dimensions: 2 X 2
# Number of Kernels: 64

up_conv_3 = layers.Conv2DTranspose(64, (2,2), (2,2), activation = 'relu', use_bias = True)(conv_17)

print("up_conv_3 completed")

print(up_conv_3.shape)


# Input tensor: (130, 130, 64)
# Output tensor: (260, 260, 64)
# Kernel Dimensions: 2 X 2
# Number of Kernels: 64

up_conv_4 = layers.Conv2DTranspose(64, (2,2), (2,2), activation = 'relu', use_bias = True)(up_conv_3)

print("up_conv_4 completed")

print(up_conv_4.shape)

# Input tensor: (260, 260, 64) & (254, 254, 64)
# Output tensor: (260, 260, 128)

resize_4_18 = layers.Resizing(260,260)(conv_1)

concat_4_18 = layers.Concatenate(axis = -1)([up_conv_4, resize_4_18])

print("concat_4_18 completed")

print(concat_4_18.shape)

# Input tensor: (260, 260, 128)
# Output tensor: (258, 258, 16)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 64

conv_18 = layers.Conv2D(16, (3,3), (1,1), activation = 'relu', use_bias = True)(concat_4_18)

print("conv_18 completed")

print(conv_18.shape)

# Input tensor: (258, 258, 16)
# Output tensor: (256, 256, 1)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 64

conv_19 = layers.Conv2D(1, (3,3), (1,1), activation = 'linear', use_bias = True)(conv_18)

print("conv_19 completed")

print(conv_19.shape)



model = keras.Model(inputs, conv_19)

model.compile(optimizer='adam',
              loss='Huber',
              metrics=['mae'])

test_input = imread(r'DIC-C2DH-HeLa\01_ST\SEG\man_seg032.tif')



test_input = np.expand_dims(test_input, axis=(0, -1))

print(test_input.shape)

test_output = model.predict(test_input[:, 0:256, 0:256])

print(test_output.shape)

imwrite("test_output.tif", test_output[:,:,:,0])

#print("Built with CUDA:", tf.test.is_built_with_cuda())

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#print(tf.__version__)

input = []

for file in os.listdir(r'DIC-C2DH-HeLa\01_ST\SEG'):
    full_path = os.path.join(r'DIC-C2DH-HeLa\01_ST\SEG', file)
    img = imread(full_path)
    for row in range(1,3):
        for col in range(1,3):
            input.append(img[((row-1) * 256): (row * 256), ((col-1) * 256): (col * 256)])

input = np.stack(input)

print(sys.getsizeof(input))

validation = []

for file in os.listdir("processed_data\cell_distance"):
    full_path = os.path.join("processed_data\cell_distance", file)
    img = imread(full_path)
    for row in range(1,3):
        for col in range(1,3):
            validation.append(img[((row-1) * 256): (row * 256), ((col-1) * 256): (col * 256)])


validation = np.stack(validation)

print("input and validation datasets made")

model.fit(input, validation, batch_size= 2,epochs=200)

model.save("image_segmentation.keras")


