import numpy as np
from tifffile import imread, imwrite
import cv2
import math
import os
from pathlib import Path

# We will create a more modular system for the model definition using python clases

def convolution (img_array, kernel, stride):
    dot_matrix = [[0 for i in range(0, math.floor((len(img_array) - len(kernel) + stride) / stride))] for j in range (0, math.floor((len(img_array) - len(kernel) + stride) / stride))]
    dot_matrix = np.asarray(dot_matrix)
    for i in range(0, len(img_array), stride):
        for j in range(0, len(img_array), stride):
            if ((i + len(kernel)) <= len(img_array)) and ((j + len(kernel)) <= len(img_array)):
                dot_matrix[i // stride, j // stride] = np.sum(img_array[i : i + len(kernel), j : j + len(kernel)] * kernel)
    #print("convolution completed")
    return dot_matrix

#up convolution function that also accounts for channel depth > 1

def up_conv (img_array, kernel, stride):
    out_dim = (len(img_array) - 1) * stride + len(kernel)
    up_img = [[0 for i in range(0, out_dim)] for j in range(0, out_dim)]
    up_img = np.asarray(up_img)
    for i in range(0, len(img_array)):
        for j in range(0, len(img_array)):
            patch = [[(np.sum(img_array[i,j] * kernel[x,y])) for x in range(0, len(kernel))] for y in range (0, len(kernel))]
            up_img[(stride * i) : ((stride * i) + len(kernel)), (stride * j) : ((stride * j) + len(kernel))] += np.asarray(patch)
    return up_img

"""
print("testing for up-conv")
test = [[2] for i in range(0, 3)]
test = np.transpose(np.asarray(test), (1,0))
print(test.shape)
test_2 = np.random.rand(3,3,3)
mult = test * test_2
print(mult.shape)
print("end of up-conv")
"""

def normalize_array(arr, min_value, max_value): 
  normalized_array = (max_value - min_value) * (arr - arr.min()) / (arr.max() - arr.min()) + min_value
  return normalized_array

def reLu (x):
    return max(0, x)

vectorize_reLu = np.vectorize(reLu)

class convBaseLayer :
    def __init__(self, kernel_size, kernel_amount, stride, input_source, layer_name):
        #Replace with .npy for more efficent computation
        self.kernel_num = kernel_amount
        self.c_stride = stride
        # Attribute created in order to backtrack
        self.input_source = input_source

        self.name = layer_name

        print(kernel_size)
        
        
        if os.path.exists(f"kernel_weights\{layer_name}.npy"):
            self.kernels = np.load(f"kernel_weights\{layer_name}.npy")
        else:
            random_kernel_gen = np.random.rand(kernel_amount, *kernel_size)
            np.save(f"kernel_weights\{layer_name}.npy", random_kernel_gen)
            self.kernels = random_kernel_gen
        
        print(self.kernels.shape)
        
        
        if os.path.exists(f"biases\{layer_name}.npy"):
            self.bias = np.load(f"biases\{layer_name}.npy")
        else:
            random_bias_gen = np.random.rand()
            np.save(f"biases\{layer_name}.npy", random_bias_gen)
            self.bias = random_bias_gen
        
    def get_input(self):
        """To be implemented by subclass"""
        raise NotImplementedError

    def transform_output(self, feature_map):
        """To be implemented by subclass"""
        raise NotImplementedError
    
    def prev_dmg_f_maps(self):

        raise NotImplementedError
    
    def conv_type (self, input_val, kern, str):

        raise NotImplementedError
    
    # I should save my out_gen from previous layers since its recusive
    def out_gen (self):
        feature_maps = [] 
        if self.prev_dmg_f_maps():
            folder_path = f"{self.input_source.name}_feature_maps"
            pre_tensor = []
            for fold in os.listdir(folder_path):
                full_path = os.path.join(folder_path, fold)
                pre_tensor.append(imread(full_path))
            input_data = np.transpose(np.stack(pre_tensor), (1,2,0))
            print(folder_path)
            print("input_data shape:")
            print(input_data.shape)
        else:
            input_data = self.get_input()
        for index in range(0,self.kernel_num):
            kernel = self.kernels[index]
            #print(self.name)
            #print(kernel.shape)
            feature_map = self.conv_type(input_data, kernel, self.c_stride)
            # Bias addition
            feature_map = feature_map + self.bias
            # ReLu activation
            feature_maps.append(self.transform_output(feature_map))
        out_tensor = np.transpose( np.stack(feature_maps), (1,2,0))
        print("out tensor shape:")
        print(out_tensor.shape)
        return out_tensor
    

class InputLayer(convBaseLayer):
    def get_input(self):
        return self.input_source
    
    def transform_output(self, feature_map):
        return vectorize_reLu(feature_map)
    
    def prev_dmg_f_maps(self):
        return False
    
    def conv_type(self, input_val, kern, str):
        return convolution(input_val, kern, str)
    
class Conv2DLayer(convBaseLayer):
    def get_input(self):
        return self.input_source.out_gen()
    
    def transform_output(self, feature_map):
        return vectorize_reLu(feature_map)
    
    def prev_dmg_f_maps(self):
        return os.path.exists(f"{self.input_source.name}_feature_maps")
    
    def conv_type(self, input_val, kern, str):
        return convolution(input_val, kern, str)
    
class Conv2DLayerPool(convBaseLayer):
    def get_input(self):
        return self.input_source.out_gen()
    
    def transform_output(self, feature_map):
        return normalize_array(vectorize_reLu(feature_map), 0, 1)
    
    def prev_dmg_f_maps(self):
        return os.path.exists(f"{self.input_source.name}_feature_maps")
    
    def conv_type(self, input_val, kern, str):
        return convolution(input_val, kern, str)

class UpConv(convBaseLayer):
    def get_input(self):
        return self.input_source.out_gen()
    
    def transform_output(self, feature_map):
        return normalize_array(vectorize_reLu(feature_map), 0, 1)
    
    def prev_dmg_f_maps(self):
        return os.path.exists(f"{self.input_source.name}_feature_maps")
    
    def conv_type(self, input_val, kern, str):
        return up_conv(input_val, kern, str)

class Concatenate(convBaseLayer):
    def __init__(self, input_sources, layer_name):
        
        self.input_sources = input_sources

        self.name = layer_name
    
    def get_input(self):
        # To account for cropping and copying, a fitted kernel convolution program can be used with a kernel filled with 1 to matain features while shriking the image
        min_len = float('inf')
        for source in self.input_sources:
            if len(source.out_gen()) < min_len:
                min_len = len(source.out_gen())
        feature_maps = []
        for i in range(0, len(self.input_sources)):
            if len(self.input_sources[i].out_gen()) > min_len:
                feature_maps.append()
                # Add a kernel based reductino scheme
        feature_maps = [source.out_gen() for source in self.input_sources]
        return np.concatenate(feature_map_save, axis = -1)
    
    def transform_output(self, feature_map):
        return feature_map
    
    def prev_dmg_f_maps(self):
        return False
    
    def conv_type(self, input_val, kern, str):
        return input_val
    
    def out_gen(self):
        output = self.get_input()
        print(f"Concatenated Output shape: {output.shape}")
        return output

def feature_map_save(layer):
    if not os.path.exists(f"{layer.name}_feature_maps"):
        os.makedirs(f"{layer.name}_feature_maps")

    output_tensor = layer.out_gen()
    for index in range(0, output_tensor.shape[2]):
        imwrite(f"{layer.name}_feature_maps\out_feature_map_{index}.tif", output_tensor[:,:,index])




def encoder_block(input_img):
    # Input tensor: (512, 512, 1)
    # Output tensor: (510, 510, 64)
    # Kernel Dimensions: 3 X 3
    # Number of Kernels: 64 

    e_m = InputLayer((3,3), 64, 1, input_img, "conv_1")

    feature_map_save(e_m)

    print("conv_1 completed")

    # Input tensor: (510, 510, 64)
    # Output tensor: (508, 508, 64)
    # Kernel Dimensions: 3 X 3 X 64
    # Number of Kernels: 64 

    e_m = Conv2DLayer((3,3, 64), 64, 1, e_m, "conv_2")

    skip_connection_1 = e_m
    
    feature_map_save(e_m)

    print("conv_2 completed")

    # Input tensor: (508, 508, 64)
    # Output tensor: (253.5 => 253, 253.5 => 253, 1)
    # Kernel Dimensions: 3 X 3 X 64
    # Number of Kernels: 1
    # Stride: 2 

    e_m = Conv2DLayerPool((3,3, 64),1,2,e_m, "pool_1")

    feature_map_save(e_m)

    print("pool_1 completed")
    
    # Input tensor: (253.5 => 253, 253.5 => 253, 1)
    # Output tensor: (251, 251, 128)
    # Kernel Dimensions: 3 X 3
    # Number of Kernels: 128
    
    e_m = Conv2DLayer((3,3), 128, 1, e_m, "conv_3")

    feature_map_save(e_m)

    print("conv_3 completed")
    
    # Input tensor: (251, 251, 128)
    # Output tensor: (249, 249, 128)
    # Kernel Dimensions: 3 X 3 X 128
    # Number of Kernels: 128
    
    e_m = Conv2DLayer((3,3, 128), 128, 1, e_m, "conv_4")

    skip_connection_2 = e_m
    
    feature_map_save(e_m)

    print("conv_4 completed")
    
    # Input tensor: (249, 249, 128)
    # Output tensor: (124, 124, 1)
    # Kernel Dimensions: 3 X 3 X 128
    # Number of Kernels: 1
    # Stride: 2
    
    e_m = Conv2DLayerPool((3,3, 128),1,2,e_m,"pool_2")

    feature_map_save(e_m)

    print("pool_2 completed")
    
    # Input tensor: (124, 124, 1)
    # Output tensor: (122, 122, 256)
    # Kernel Dimensions: 3 X 3
    # Number of Kernels: 256
    
    e_m = Conv2DLayer((3,3), 256, 1, e_m ,"conv_5")

    feature_map_save(e_m)

    print("conv_5 completed")
    
    # Input tensor: (122, 122, 256)
    # Output tensor: (120, 120, 256)
    # Kernel Dimensions: 3 X 3 X 256
    # Number of Kernels: 256
    
    e_m = Conv2DLayer((3,3, 256),256, 1, e_m, "conv_6")

    skip_connection_3 = e_m
    
    feature_map_save(e_m)

    print("conv_6 completed")
    
    # Input tensor: (120, 120, 256)
    # Output tensor: (59.5 => 59, 59.5 => 59, 1)
    # Kernel Dimensions: 3 X 3 X 256
    # Number of Kernels: 1
    # stride: 2
    
    e_m = Conv2DLayerPool((3,3, 256),1,2,e_m,"pool_3")

    feature_map_save(e_m)

    print("pool_3 completed")
    
    # Input tensor: (59.5 => 59, 59.5 => 59, 1)
    # Output tensor: (57,57, 512)
    # Kernel Dimensions: 3 X 3
    # Number of Kernels: 512
    
    e_m = Conv2DLayer((3,3),512, 1, e_m, "conv_7")

    feature_map_save(e_m)

    print("conv_7 completed")
    
    # Input tensor: (57,57, 512)
    # Output tensor: (55,55, 512)
    # Kernel Dimensions: 3 X 3 X 512
    # Number of Kernels: 512
    
    e_m = Conv2DLayer((3,3, 512), 512, 1, e_m, "conv_8")

    skip_connection_4 = e_m
    
    feature_map_save(e_m)

    print("conv_8 completed")
    
    # Input tensor: (55,55, 512)
    # Output tensor: (27,27, 1)
    # Kernel Dimensions: 3 X 3 X 512
    # Number of Kernels: 1
    # Stride: 2
    
    e_m = Conv2DLayerPool((3,3, 512), 1, 2, e_m, "pool_4")

    feature_map_save(e_m)

    print("pool_4 completed")
    
    # Input tensor: (27,27, 1)
    # Output tensor: (25,25, 1024)
    # Kernel Dimensions: 3 X 3
    # Number of Kernels: 1024
    
    e_m = Conv2DLayer((3,3), 1024, 1, e_m, "conv_9")

    feature_map_save(e_m)

    print("conv_9 completed")
    
    # Input tensor: (25,25, 1024)
    # Output tensor: (23,23, 1024)
    # Kernel Dimensions: 3 X 3 X 1024
    # Number of Kernels: 1024
    
    e_m = Conv2DLayer((3, 3, 1024), 1024, 1, e_m, "conv_10")

    feature_map_save(e_m)

    print("conv_10 completed")

    # Input tensor: (23,23, 1024)
    # Output tensor: (46, 46, 1024)
    # Kernel Dimensions: 2 X 2 X 1024
    # Number of Kernels: 1024
    # Stride: 2
    
    e_m = UpConv((2,2, 1024), 1024, 2, e_m, "up_conv_1")

    feature_map_save(e_m)

    print("up_conv_1 completed")

    # Input tensor: (46, 46, 1024)
    # Output tensor: (44, 44, 512)
    # Kernel Dimensions: 3 X 3 X 1024
    # Number of Kernels: 512
    
    e_m = Conv2DLayer((3,3,1024), 512, 1, e_m, "conv_11")

    feature_map_save(e_m)

    print("conv_11 completed")

    # Input tensor: (44, 44, 512)
    # Output tensor: (42, 42, 512)
    # Kernel Dimensions: 3 X 3 X 512
    # Number of Kernels: 512
    
    e_m = Conv2DLayer((3,3,512), 512, 1, e_m, "conv_12")

    feature_map_save(e_m)

    print("conv_12 completed")

    #Concatenate a skip connection to produce an out_tensor with channel depth of 512
    # Input tensor: (42, 42, 512)
    # Output tensor: (84, 84, 256)
    # Kernel Dimensions: 2 X 2 X 512
    # Number of Kernels: 512
    
    e_m = UpConv((2,2,512), 256, 2, e_m, "up_conv_2")

    feature_map_save(e_m)

    print("up_conv_2 completed")

    # Input tensor: (84, 84, 512)
    # Output tensor: (82, 82, 256)
    # Kernel Dimensions: 3 X 3 X 512
    # Number of Kernels: 256
    
    e_m = Conv2DLayer((3,3,512), 256, 1, e_m, "conv_13")

    feature_map_save(e_m)

    print("conv_13 completed")

    # Input tensor: (82, 82, 256)
    # Output tensor: (80, 80, 256)
    # Kernel Dimensions: 3 X 3 X 256
    # Number of Kernels: 256
    
    e_m = Conv2DLayer((3,3,256), 256, 1, e_m, "conv_14")

    feature_map_save(e_m)

    print("conv_14 completed")

    # Concatenate a skip connection to create an out tensor of channel depth 256
    # Input tensor: (80, 80, 256)
    # Output tensor: (160, 160, 128)
    # Kernel Dimensions: 2 X 2 X 256
    # Number of Kernels: 128
    # Stride: 2
    
    
    e_m = UpConv((2,2, 256), 128, 2, e_m, "up_conv_3")

    feature_map_save(e_m)

    print("up_conv_3 completed")

    # Input tensor: (160, 160, 256)
    # Output tensor: (158, 158, 198)
    # Kernel Dimensions: 3 X 3 X 256
    # Number of Kernels: 198
    
    e_m = Conv2DLayer((3,3,256), 198, 1, e_m, "conv_15")

    feature_map_save(e_m)

    print("conv_15 completed")

    # Input tensor: (158, 158, 198)
    # Output tensor: (156, 156, 128)
    # Kernel Dimensions: 3 X 3 X 198
    # Number of Kernels: 128
    
    e_m = Conv2DLayer((3,3,198), 128, 1, e_m, "conv_16")

    feature_map_save(e_m)

    print("conv_16 completed")

    # Concatenate a skip connection to create an out tensor with channel depth 128
    # Input tensor: (156, 156, 128)
    # Output tensor: (312, 312, 64)
    # Kernel Dimensions: 3 X 3 X 128
    # Number of Kernels: 64
    # Stride: 2
    
    e_m = UpConv((2,2, 128), 64, 2, e_m, "up_conv_4")

    feature_map_save(e_m)

    print("up_conv_4 completed")

    # Input tensor: (312, 312, 64)
    # Output tensor: (284, 284, 64)
    # Kernel Dimensions: 29 X 29 X 64
    # Number of Kernels: 64
    
    e_m = Conv2DLayer((29,29,64), 64, 1, e_m, "conv_17")

    feature_map_save(e_m)

    print("conv_17 completed")

    # Input tensor: (284, 284, 64)
    # Output tensor: (258, 258, 64)
    # Kernel Dimensions: 27 X 27 X 64
    # Number of Kernels: 64
    
    e_m = Conv2DLayer((27,27,64), 64, 1, e_m, "conv_18")

    feature_map_save(e_m)

    print("conv_18 completed")

    # Input tensor: (258, 258, 64)
    # Output tensor: (256, 256, 64)
    # Kernel Dimensions: 3 X 3 X 64
    # Number of Kernels: 64
    
    e_m = Conv2DLayer((3,3,64), 64, 1, e_m, "conv_19")

    feature_map_save(e_m)

    print("conv_19 completed")

    # Input tensor: (256, 256, 64)
    # Output tensor: (512, 512, 64)
    # Kernel Dimensions: 3 X 3 X 64
    # Number of Kernels: 64
    # Stride: 2

    e_m = UpConv((2,2, 64), 64, 2, e_m, "up_conv_5")

    feature_map_save(e_m)

    print("up_conv_5 completed")

    # Input tensor: (512, 512, 64)
    # Output tensor: (512, 512, 2)
    # Kernel Dimensions: 1 X 1 X 64
    # Number of Kernels: 2
    # Stride: 1
    
    e_m = Conv2DLayer((1,1,64),2, 1, e_m, "conv_20")

    feature_map_save(e_m)

    print("conv_20 completed")



input_image = r"DIC-C2DH-HeLa\01\t014.tif"

#The cell tracking model will be an adaption of the UNet architecture with some modifications. 



image = imread(input_image)

image_array = np.asarray(image)


curr_model = encoder_block(image_array)
