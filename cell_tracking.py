import numpy as np
from tifffile import imread, imwrite
import cv2
import math

input_image = r"DIC-C2DH-HeLa\01\t014.tif"

#The cell tracking model will be an adaption of the UNet architecture with some modifications. 

image = imread(input_image)

image_array = np.asarray(image)

test1 = np.random.rand(3,3, 64)

test2 = np.random.rand(5,5, 64)

print(np.sum(test1 * test2[:3, :3]))

def convolution (img_array, kernel, stride):
    dot_matrix = [[0 for i in range(0, math.floor((len(img_array) - len(kernel) + stride) / stride))] for j in range (0, math.floor((len(img_array) - len(kernel) + stride) / stride))]
    dot_matrix = np.asarray(dot_matrix)
    for i in range(0, len(img_array), stride):
        for j in range(0, len(img_array), stride):
            if ((i + len(kernel)) <= len(img_array)) and ((j + len(kernel)) <= len(img_array)):
                dot_matrix[i // stride, j // stride] = np.sum(img_array[i : i + len(kernel), j : j + len(kernel)] * kernel)
    print("convolution completed")
    return dot_matrix

def normalize_array(arr, min_value, max_value): 
  normalized_array = (max_value - min_value) * (arr - arr.min()) / (arr.max() - arr.min()) + min_value
  return normalized_array

def reLu (x):
    return max(0, x)

sobel_filter = np.array([[1,0,-1],
                        [2,0,-2],
                        [1,0,-1]])

sobel_kernel = np.stack([sobel_filter] * 64)

sobel_kernel = np.transpose(sobel_kernel, (1,2,0))

vectorize_reLu = np.vectorize(reLu)

print(sobel_kernel.shape)
#First convolutional layer with a 3x3 kernel and ReLu acitvation function. 
# Input tensor: (1024,1024,1). 
# Output tensor: (1022,1022, 64)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 64


c1_kernels = [np.random.rand(3, 3) for _ in range(64)]

feature_maps_1 = []

b1 =  np.random.rand()

for kernel in c1_kernels:
    feature_map = convolution(image_array, kernel, 1)
    # Bias addition
    feature_map = feature_map + b1
    # ReLu activation
    feature_maps_1.append(vectorize_reLu(feature_map))

# np.stack combines 2D array into a 3D tensor
convolution_l_1 = np.stack(feature_maps_1)

#np.trasnpose trasnforms the dimensions of the tensor from (64, 1022, 1022) to (1022, 1022, 64) 
convolution_l_1 = np.transpose(convolution_l_1, (1,2,0))


imwrite('c1_feature_map.tif', convolution_l_1[:,:,0])
print(convolution_l_1.shape)


#Second convolutional layer with a 3x3 kernal and ReLu activation function
# Input tensor: (1022, 1022, 64)
# Output tensor: (1020, 1020, 64)
# Kernel Dimensions: 64 X 3 X 3
# Number of Kernels: 64 
c2_kernels = [np.random.rand(3,3, 64) for _ in range(64)]

convolution_l_2 = []

feature_maps_2 = []

b2 = np.random.rand()

for kernel in c2_kernels:
    feature_map = convolution(convolution_l_1, kernel, 1)
    # Bias addition
    feature_map = feature_map + b2
    # ReLu activation
    feature_maps_2.append(vectorize_reLu(feature_map))

# np.stack combines 2D array into a 3D tensor
convolution_l_2= np.stack(feature_maps_2)

#np.trasnpose trasnforms the dimensions of the tensor from (64, 1020, 1020) to (1020, 1020, 64) 
convolution_l_2 = np.transpose(convolution_l_2, (1,2,0))

imwrite('c2_feature_map.tif', convolution_l_2[:,:,0])
print(convolution_l_2.shape)



# This layer is technically a max pooling layer in UNet, but it will be replace by a 2D convolutional layer with stride 2 and kernel 3
# Input tensor: (1020, 1020, 64)
# Output tensor: (510, 510, 64)
# Kernel Dimensions: 64 x 3 x 3
# Number of Kernels: 1
# Stride: 2
m1_kernels = [np.random.rand(3,3, 64) for _ in range(64)]

convolution_l_3 = []

feature_maps_3 = []

b3 = np.random.rand()


for kernel in m1_kernels:
    feature_map = convolution(convolution_l_2, kernel, 2)
    # Bias addition
    feature_map = feature_map + b3
    # ReLu activation
    feature_maps_3.append(normalize_array(vectorize_reLu(feature_map), 0, 1))

# np.stack combines 2D array into a 3D tensor
convolution_l_3= np.stack(feature_maps_3)

#np.trasnpose transforms the dimensions of the tensor from (64, 1020, 1020) to (1020, 1020, 64) 
convolution_l_3 = np.transpose(convolution_l_3, (1,2,0))

for index in range(0, convolution_l_3.shape[2]):
    imwrite(f'pooled_f1_maps\pooled_feature_map_{index}.tif', convolution_l_3[:,:,index])

print(convolution_l_3.shape)

#Third convolutional layer with a 3x3 kernel and ReLu activation function
# Input tensor: (510, 510, 1)
# Output tensor: (508, 508, 128)
# Kernel Dimensions: 3 X 3
# Number of Kernels: 128
c3_kernel = np.random.rand(3,3)

#Fourth convolutional layer with a 3x3 kernel and ReLu activation function
# Input tensor: (508, 508, 128)
# Output tensor: (506, 506, 128)
# Kernel Dimensions: 128 X 3 X 3
# Number of Kernels: 128
c4_kernel = np.random.rand(3,3)

# This layer is technically a max pooling layer in UNet, but it will be replace by a 2D convolutional layer with stride 2 and kernel 3
# Input tensor: (506, 506, 128)
# Output tensor: (203, 203, 1)
# Kernel Dimensions: 128 X 3 X 3
# Number of Kernels: 1
# Stride : 2
m2_kernel = np.random.rand(3,3)

#Fifth convolutional layer with a 3x3 kernel and ReLu activation function
# Input tensor: (203, 203, 1)
# Output tensor: (200, 200, 256)
# Kernel Dimensions: 4 X 4
# Number of Kernels: 256
c3_kernel = np.random.rand(3,3)

#Sixth convolutional layer with a 3x3 kernel and ReLu activation function
# Input tensor: (200, 200, 256)
# Output tensor: (198, 198, 256)
# Kernel Dimensions: 256 X 3 X 3
# Number of Kernels: 256 
c4_kernel = np.random.rand(3,3)