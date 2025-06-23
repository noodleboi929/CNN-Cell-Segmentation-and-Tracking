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



# I will define the model using a Functional API

inputs = keras.Input(shape = (512,512, 1))

# Input tensor: (512, 512, 1)
# Output tensor: (510, 510,
#  64)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 64 

conv_1 = layers.Conv2D(64, (3,3), (1,1), activation= 'relu', use_bias = True)(inputs)

print("conv_1 completed")

print(conv_1.shape)

# Input tensor: (510, 510, 64)
# Output tensor: (508, 508, 64)
# Kernel Dimensions: 3 X 3 X 64
# Number of Kernels: 64

conv_2 = layers.Conv2D(64, (3,3), (1,1), activation = 'relu', use_bias = True)(conv_1)

#skip_connection_1 = conv_2.output

print("conv_2 completed")

print(conv_2.shape)

# Input tensor: (508, 508, 64)
# Output tensor: (253.5 => 253, 253.5 => 253, 1)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 1
# Stride: 2 

conv_pool_1 = layers.AveragePooling2D((3,3), (2,2))(conv_2)

print("conv_pool_1 completed")

print(conv_pool_1.shape)

# Input tensor: (253.5 => 253, 253.5 => 253, 1)
# Output tensor: (251, 251, 128)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 128

conv_3 = layers.Conv2D(128, (3,3), (1,1), activation = 'relu', use_bias = True)(conv_pool_1)

print("conv_3 completed")

print(conv_3.shape)

# Input tensor: (251, 251, 128)
# Output tensor: (249, 249, 128)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 128

conv_4 = layers.Conv2D(128, (3,3), (1,1), activation = 'relu', use_bias = True)(conv_3)

# skip_connection_2 = conv_4.output

print("conv_4 completed")

print(conv_4.shape)

# Input tensor: (249, 249, 128)
# Output tensor: (124, 124, 1)
# Kernel Dimensions: 3 X 3 X 128
# Number of Kernels: 1
# Stride: 2

conv_pool_layer_2 = layers.AveragePooling2D((3,3), (2,2))(conv_4)

print("conv_pool_layer_2 completed")

print(conv_pool_layer_2.shape)

# Input tensor: (124, 124, 1)
# Output tensor: (122, 122, 256)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 256

conv_5 = layers.Conv2D(256, (3,3), (1,1), activation = 'relu', use_bias = True)(conv_pool_layer_2)

print("conv_5 completed")

print(conv_5.shape)

# Input tensor: (122, 122, 256)
# Output tensor: (120, 120, 256)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 256

conv_6 = layers.Conv2D(256, (3,3), (1,1), activation = 'relu', use_bias = True)(conv_5)

#skip_connection_3 = conv_6.output

print("conv_6 completed")

print(conv_6.shape)

# Input tensor: (120, 120, 256)
# Output tensor: (59.5 => 59, 59.5 => 59, 1)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 1
# stride: 2

conv_pool_3 = layers.AveragePooling2D((3,3), (2,2))(conv_6)

print("conv_pool_3 completed")

print(conv_pool_3.shape)

# Input tensor: (59.5 => 59, 59.5 => 59, 1)
# Output tensor: (57,57, 512)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 512

conv_7 = layers.Conv2D(512, (3,3), (1,1), activation= 'relu', use_bias = True)(conv_pool_3)

print("conv_7 completed")

print(conv_7.shape)

# Input tensor: (57,57, 512)
# Output tensor: (55,55, 512)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 512

conv_8 = layers.Conv2D(512, (3,3), (1,1), activation = 'relu', use_bias = True)(conv_7)

# skip_connection_4 = conv_8.output

print("Conv_8 completed")

print(conv_8.shape)

# Input tensor: (55,55, 512)
# Output tensor: (27,27, 1)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 1
# Stride: 2

conv_pool_4 = layers.AveragePooling2D((3,3), (2,2))(conv_8)

print("conv_pool_4 completed")

print(conv_pool_4.shape)

# Input tensor: (27,27, 1)
# Output tensor: (25,25, 1024)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 1024

conv_9 = layers.Conv2D(1024, (3,3), (1,1), activation = 'relu', use_bias = True)(conv_pool_4)

print("conv_9 completed")

print(conv_9.shape)

# Input tensor: (25,25, 1024)
# Output tensor: (23,23, 1024)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 1024

conv_10 = layers.Conv2D(1024, (3,3), (1,1), activation = 'relu', use_bias = True)(conv_9)

print("conv_10 completed")

print(conv_10.shape)

# Input tensor: (23,23, 1024)
# Output tensor: (46, 46, 512)
# Kernel Dimensions: 2 X 2
# Number of Kernels: 512
# Stride: 2

up_conv_1 = layers.Conv2DTranspose(512, (2,2), (2,2), activation = 'relu', use_bias = True)(conv_10)

print("up_conv_1 completed")

print(up_conv_1.shape)

# Input tensor: (46, 46, 512) & (55, 55, 512)
# Output tensor: (46,46, 1024)

resize_4 = layers.Resizing(46,46)(conv_8)

concat_1 = layers.Concatenate(axis = -1)([up_conv_1, resize_4])

print("concat_1 completed")

print(concat_1.shape)

# Input tensor: (46, 46, 1024)
# Output tensor: (44, 44, 512)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 512

conv_11 = layers.Conv2D(512, (3,3), (1,1), activation = 'relu', use_bias = True)(concat_1)

print("conv_11 completed")

print(conv_11.shape)

# Input tensor: (44, 44, 512)
# Output tensor: (42, 42, 512)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 512

conv_12 = layers.Conv2D(512, (3,3), (1,1), activation = 'relu', use_bias = True)(conv_11)

print("conv_12 completed")

print(conv_12.shape)

# Concatenate a skip connection to produce an out_tensor with channel depth of 512
# Input tensor: (42, 42, 512)
# Output tensor: (84, 84, 256)
# Kernel Dimensions: 2 X 2
# Number of Kernels: 512

up_conv_2 = layers.Conv2DTranspose(512, (2,2), (2,2), activation = 'relu', use_bias = True)(conv_12)

print("up_conv02 completed")

print(up_conv_2.shape)

# Input tensor: (84, 84, 256) & (120, 120, 256)
# Output tensor: (84,84, 512)

resize_3 = layers.Resizing(84,84)(conv_6)

concat_2 = layers.Concatenate(axis = -1)([up_conv_2, resize_3])

print("concate_2 completed")

print(concat_2.shape)

# Input tensor: (84, 84, 512)
# Output tensor: (82, 82, 256)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 256

conv_13 = layers.Conv2D(256, (3,3), (1,1), activation = 'relu', use_bias = True)(concat_2)

print("conv_13 completed")

print(conv_13.shape)

# Input tensor: (82, 82, 256)
# Output tensor: (80, 80, 256)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 256

conv_14 = layers.Conv2D(256, (3,3), (1,1), activation = 'relu', use_bias = True)(conv_13)

print("conv_14 completed")

print(conv_14.shape)

# Concatenate a skip connection to create an out tensor of channel depth 256
# Input tensor: (80, 80, 256)
# Output tensor: (160, 160, 128)
# Kernel Dimensions: 2 X 2
# Number of Kernels: 128
# Stride: 2

up_conv_3 = layers.Conv2DTranspose(128, (2,2), (2,2), activation = 'relu', use_bias = True)(conv_14)

print("up_conv_3 completed")

print(up_conv_3.shape)

# Input tensor: (160, 160, 128) & (249, 249, 128)
# Output tensor: (160, 160, 256)

resize_2 = layers.Resizing(160,160)(conv_4)

concat_3 = layers.Concatenate(axis = -1)([up_conv_3, resize_2])

print("concat_3 completed")

print(concat_3.shape)

# Input tensor: (160, 160, 256)
# Output tensor: (158, 158, 198)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 198

conv_15 = layers.Conv2D(198, (3,3), (1,1), activation = 'relu', use_bias = True)(concat_3)

print("conv_15 completed")

print(conv_15.shape)

# Input tensor: (158, 158, 198)
# Output tensor: (156, 156, 128)
# Kernel Dimensions: 3 X 3 X 198
# Number of Kernels: 128

conv_16 = layers.Conv2D(128, (3,3), (1,1), activation = 'relu', use_bias = True)(conv_15)

print("conv_16 completed")

print(conv_16.shape)

# Concatenate a skip connection to create an out tensor with channel depth 128
# Input tensor: (156, 156, 128)
# Output tensor: (312, 312, 64)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 64
# Stride: 2

up_conv_4 = layers.Conv2DTranspose(64, (3,3), (2,2), activation = 'relu', use_bias = True)(conv_16)

print("up_conv_4 completed")

print(up_conv_4.shape)

# Input tensor: (312, 312, 64) & (508, 508, 64)
# Output tensor: (312, 312, 128)

resize_1 = layers.Resizing(312,312)(conv_2)

resize_up = layers.Resizing(312,312)(up_conv_4)

concat_4 = layers.Concatenate(axis = -1)([resize_up, resize_1])

print("concat_4 completed")

print(concat_4.shape)

# Input tensor: (312, 312, 128)
# Output tensor: (284, 284, 64)
# Kernel Dimensions: 29 X 29
# Number of Kernels: 64

conv_17 = layers.Conv2D(64, (29,29), (1,1), activation = 'relu', use_bias = True)(concat_4)

print("conv_17 completed")

print(conv_17.shape)

# Input tensor: (284, 284, 64)
# Output tensor: (258, 258, 64)
# Kernel Dimensions: 27 X 27
# Number of Kernels: 64

conv_18 = layers.Conv2D(64, (27,27), (1,1), activation = 'relu', use_bias = True)(conv_17)

print("conv_18 completed")

print(conv_18.shape)

# Input tensor: (258, 258, 64)
# Output tensor: (256, 256, 64)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 64

conv_19 = layers.Conv2D(64, (3,3), (1,1), activation = 'relu', use_bias = True)(conv_18)

print("conv_19 completed")

print(conv_19.shape)

# Input tensor: (256, 256, 64)
# Output tensor: (516, 516, 64)
# Kernel Dimensions: 2 X 2
# Number of Kernels: 64
# Stride: 2

up_conv_5 = layers.Conv2DTranspose(64, (2,2), (2,2), activation = 'relu', use_bias = True)(conv_18)

print("up_conv_5 completed")

print(up_conv_5.shape)

# Input tensor: (512, 512, 64)
# Output tensor: (512, 512, 2)
# Kernel Dimensions: 5 X 5
# Number of Kernels: 2
# Stride: 1

conv_20 = layers.Conv2D(1, (5,5), (1,1), activation = 'relu', use_bias = True)(up_conv_5)

print("conv_20 completed")

print(conv_20.shape)

model = keras.Model(inputs, conv_20)

model.compile(optimizer='adam',
              loss='Huber',
              metrics=['accuracy'])

test_input = imread(r'DIC-C2DH-HeLa\01_ST\SEG\man_seg032.tif')

test_input = test_input.astype(np.float32) / 255.0

test_input = np.expand_dims(test_input, axis=(0, -1))

print(test_input.shape)

test_output = model.predict(test_input)

print(test_output.shape)

imwrite("test_output.tif", test_output[:,:,:,0])

#print("Built with CUDA:", tf.test.is_built_with_cuda())

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#print(tf.__version__)

input = []

for file in os.listdir(r'DIC-C2DH-HeLa\01_ST\SEG'):
    full_path = os.path.join(r'DIC-C2DH-HeLa\01_ST\SEG', file)
    input.append(imread(full_path))

input = np.stack(input)

print(sys.getsizeof(input))

validation = []

for file in os.listdir("processed_data\cell_distance"):
    full_path = os.path.join("processed_data\cell_distance", file)
    validation.append(imread(full_path))

validation = np.stack(validation)

print("input and validation datasets made")

model.fit(input, validation, batch_size= 1,epochs=20)

model.save("image_segmentation.keras")


