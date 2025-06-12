import numpy as np
from tifffile import imread, imwrite
import cv2
import math
import os
from pathlib import Path

# We will create a more modular system for the model definition using python clases

def convolution (img_array, kernel, stride):
    # dot_matrix = [[0 for i in range(0, math.floor((len(img_array) - len(kernel) + stride) / stride))] for j in range (0, math.floor((len(img_array) - len(kernel) + stride) / stride))]
    # dot_matrix = np.asarray(dot_matrix)
    dot_matrix = np.zeros((math.floor((len(img_array) - len(kernel) + stride) / stride), math.floor((len(img_array) - len(kernel) + stride) / stride)), dtype= np.float32)
    for i in range(0, len(img_array), stride):
        for j in range(0, len(img_array), stride):
            if ((i + len(kernel)) <= len(img_array)) and ((j + len(kernel)) <= len(img_array)):
                dot_matrix[i // stride, j // stride] = np.sum(img_array[i : i + len(kernel), j : j + len(kernel)] * kernel, dtype= np.float32)
    #print("convolution completed")
    return dot_matrix

def convolution_w_reshape (kernels, img_array, stride, kernel_shape, kernel_num):
    reshape_array = []
    kernel_dim = kernel_shape[0]
    img_dim = len(img_array)
    final_dim = math.floor((img_dim - kernel_dim + stride) / stride)
    for i in range(0, img_dim, stride):
        for j in range(0, img_dim, stride):
            if ((i + kernel_dim) <= img_dim) and ((j + kernel_dim) <= img_dim):
                patch = img_array[i : i + kernel_dim, j : j + kernel_dim]
                tiled_patch = np.tile(patch, (kernel_num,) + tuple(1 for i in range(0, patch.ndim)))
                reshape_array.append(tiled_patch)
    
    reshape_array = (np.stack(reshape_array))
    print(f"reshape array dim: {reshape_array.shape}") 
    dot_matrix = (kernels[None, ...] *  reshape_array)
    dot_matrix = np.sum(dot_matrix, axis=tuple(range(2, dot_matrix.ndim)), dtype= np.float16)
    print(f"pre-shaped dim: {dot_matrix.shape}")
    print(f"Dimension Check: {dot_matrix.shape[0] == (final_dim * final_dim)}")
    dot_matrix = np.reshape(dot_matrix, (kernel_num, final_dim, final_dim), order= "C")
    dot_matrix = np.transpose(dot_matrix, (1,2,0))
    return dot_matrix



test_kernel = np.random.rand(64,3,3,64)

test_img_array = np.random.rand(512,512,64)
print("Original Convolution:")

for index in range(0, 64):
    print(index)
    O_conv = convolution(test_img_array, test_kernel[index], 1)

print(O_conv)

print("Convolution with reshape:")

new_conv = convolution_w_reshape(test_kernel, test_img_array, 1, (3,3,9), 64)

print(new_conv[...,63])

print("Equality_check:")

print(f"{np.array_equal(O_conv,new_conv)}")