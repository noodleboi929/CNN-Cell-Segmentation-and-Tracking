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

#up convolution function that also accounts for channel depth > 1

def up_conv (img_array, kernel, stride):
    out_dim = (len(img_array) - 1) * stride + len(kernel)
    up_img = np.zeros((out_dim, out_dim), dtype= np.float64)
    for i in range(0, len(img_array)):
        for j in range(0, len(img_array)):
            patch = [[(np.sum(img_array[i,j] * kernel[x,y])) for x in range(0, len(kernel))] for y in range (0, len(kernel))]
            up_img[(stride * i) : ((stride * i) + len(kernel)), (stride * j) : ((stride * j) + len(kernel))] += np.asarray(patch)
    return up_img


def shrink (img_array, shrink_val):
    H, W = img_array.shape[:2]
    if len(img_array.shape) > 2:
        C = img_array.shape[2]
        shrunk_img = np.zeros(((H - shrink_val), (W - shrink_val),C))
    else:
        C = 1
        shrunk_img = np.zeros(((H - shrink_val), (W - shrink_val)))
    kernel= np.ones((shrink_val + 1, shrink_val + 1))
    for i in range(0, len(img_array)):
        for j in range(0, len(img_array)):
            if ((i + len(kernel)) <= len(img_array)) and ((j + len(kernel)) <= len(img_array)):
                if C > 1:
                    for c in range(0, C):
                        shrunk_img[i,j, c] = np.sum(img_array[i : i + len(kernel), j : j + len(kernel), c] * kernel)
                else:
                    shrunk_img[i,j] = np.sum(img_array[i : i + len(kernel), j : j + len(kernel)] * kernel)
    return shrunk_img

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

def derv_relu (x):
    if x < 0 :
        return 0
    return 1

vectorize_derv_relu = np.vectorize(derv_relu)

vectorize_reLu = np.vectorize(reLu)

class convBaseLayer :
    def __init__(self, kernel_size, kernel_amount, stride, input_source, layer_name, load_bool):
        #Replace with .npy for more efficent computation
        self.kernel_num = kernel_amount
        self.c_stride = stride
        # Attribute created in order to backtrack
        self.input_source = input_source

        self.name = layer_name

        self.load_val = load_bool

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
        if self.load_val and os.path.exists(f"{self.name}_feature_maps") and (len(os.listdir(f"{self.name}_feature_maps")) > 0):
            load_tensor = []
            for file in os.listdir(f"{self.name}_feature_maps"):
                full_path = os.path.join(f"{self.name}_feature_maps", file)
                load_tensor.append(imread(full_path))
            load_tensor = np.transpose(np.stack(load_tensor), (1,2,0))
            print("loaded_tensor: ")
            print(load_tensor.shape)
            return load_tensor

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
    def __init__(self, input_sources, layer_name, load_bool):
        
        self.input_sources = input_sources

        self.name = layer_name

        self.load_val = load_bool
    
    def get_input(self):
        # To account for cropping and copying, a fitted kernel convolution program can be used with a kernel filled with 1 to matain features while shriking the image
        min_len = float('inf')
        for source in self.input_sources:
            if len(source.out_gen()) < min_len:
                min_len = len(source.out_gen())
        feature_maps = []
        for i in range(0, len(self.input_sources)):
            if len(self.input_sources[i].out_gen()) > min_len:
                feature_maps.append(shrink(self.input_sources[i].out_gen(), len(self.input_sources[i].out_gen()) - min_len))
                # Add a kernel based reductino scheme
            else:
                feature_maps.append(self.input_sources[i].out_gen())
        return np.concatenate(feature_maps, axis = -1)
    
    def transform_output(self, feature_map):
        return normalize_array(vectorize_reLu(feature_map), 0, 1)
    
    def prev_dmg_f_maps(self):
        return False
    
    def conv_type(self, input_val, kern, str):
        return input_val
    
    def out_gen(self):
        if self.load_val and os.path.exists(f"{self.name}_feature_maps") and (len(os.listdir(f"{self.name}_feature_maps")) > 0):
            load_tensor = []
            for file in os.listdir(f"{self.name}_feature_maps"):
                full_path = os.path.join(f"{self.name}_feature_maps", file)
                load_tensor.append(imread(full_path))
            load_tensor = np.transpose(np.stack(load_tensor), (1,2,0))
            print("loaded_tensor: ")
            print(load_tensor.shape)
            return load_tensor
        output = self.get_input()
        print(f"Concatenated Output shape: {output.shape}")
        return output

def feature_map_save(layer):
    if not os.path.exists(f"{layer.name}_feature_maps"):
        os.makedirs(f"{layer.name}_feature_maps")

    output_tensor = layer.out_gen()
    for index in range(0, output_tensor.shape[2]):
        imwrite(f"{layer.name}_feature_maps\out_feature_map_{index}.tif", output_tensor[:,:,index])




def encoder_block(input_img, load_bool):
    # Input tensor: (512, 512, 1)
    # Output tensor: (510, 510, 64)
    # Kernel Dimensions: 3 X 3
    # Number of Kernels: 64 

    e_m = InputLayer((3,3), 64, 1, input_img, "conv_1", load_bool)

    feature_map_save(e_m)

    print("conv_1 completed")

    # Input tensor: (510, 510, 64)
    # Output tensor: (508, 508, 64)
    # Kernel Dimensions: 3 X 3 X 64
    # Number of Kernels: 64 

    e_m = Conv2DLayer((3,3, 64), 64, 1, e_m, "conv_2", load_bool)

    skip_connection_1 = e_m
    
    feature_map_save(e_m)

    print("conv_2 completed")

    # Input tensor: (508, 508, 64)
    # Output tensor: (253.5 => 253, 253.5 => 253, 1)
    # Kernel Dimensions: 3 X 3 X 64
    # Number of Kernels: 1
    # Stride: 2 

    e_m = Conv2DLayerPool((3,3, 64),1,2,e_m, "pool_1", load_bool)

    feature_map_save(e_m)

    print("pool_1 completed")
    
    # Input tensor: (253.5 => 253, 253.5 => 253, 1)
    # Output tensor: (251, 251, 128)
    # Kernel Dimensions: 3 X 3
    # Number of Kernels: 128
    
    e_m = Conv2DLayer((3,3), 128, 1, e_m, "conv_3", load_bool)

    feature_map_save(e_m)

    print("conv_3 completed")
    
    # Input tensor: (251, 251, 128)
    # Output tensor: (249, 249, 128)
    # Kernel Dimensions: 3 X 3 X 128
    # Number of Kernels: 128
    
    e_m = Conv2DLayer((3,3, 128), 128, 1, e_m, "conv_4", load_bool)

    skip_connection_2 = e_m
    
    feature_map_save(e_m)

    print("conv_4 completed")
    
    # Input tensor: (249, 249, 128)
    # Output tensor: (124, 124, 1)
    # Kernel Dimensions: 3 X 3 X 128
    # Number of Kernels: 1
    # Stride: 2
    
    e_m = Conv2DLayerPool((3,3, 128),1,2,e_m,"pool_2", load_bool)

    feature_map_save(e_m)

    print("pool_2 completed")
    
    # Input tensor: (124, 124, 1)
    # Output tensor: (122, 122, 256)
    # Kernel Dimensions: 3 X 3
    # Number of Kernels: 256
    
    e_m = Conv2DLayer((3,3), 256, 1, e_m ,"conv_5", load_bool)

    feature_map_save(e_m)

    print("conv_5 completed")
    
    # Input tensor: (122, 122, 256)
    # Output tensor: (120, 120, 256)
    # Kernel Dimensions: 3 X 3 X 256
    # Number of Kernels: 256
    
    e_m = Conv2DLayer((3,3, 256),256, 1, e_m, "conv_6", load_bool)

    skip_connection_3 = e_m
    
    feature_map_save(e_m)

    print("conv_6 completed")
    
    # Input tensor: (120, 120, 256)
    # Output tensor: (59.5 => 59, 59.5 => 59, 1)
    # Kernel Dimensions: 3 X 3 X 256
    # Number of Kernels: 1
    # stride: 2
    
    e_m = Conv2DLayerPool((3,3, 256),1,2,e_m,"pool_3", load_bool)

    feature_map_save(e_m)

    print("pool_3 completed")
    
    # Input tensor: (59.5 => 59, 59.5 => 59, 1)
    # Output tensor: (57,57, 512)
    # Kernel Dimensions: 3 X 3
    # Number of Kernels: 512
    
    e_m = Conv2DLayer((3,3),512, 1, e_m, "conv_7", load_bool)

    feature_map_save(e_m)

    print("conv_7 completed")
    
    # Input tensor: (57,57, 512)
    # Output tensor: (55,55, 512)
    # Kernel Dimensions: 3 X 3 X 512
    # Number of Kernels: 512
    
    e_m = Conv2DLayer((3,3, 512), 512, 1, e_m, "conv_8", load_bool)

    skip_connection_4 = e_m
    
    feature_map_save(e_m)

    print("conv_8 completed")
    
    # Input tensor: (55,55, 512)
    # Output tensor: (27,27, 1)
    # Kernel Dimensions: 3 X 3 X 512
    # Number of Kernels: 1
    # Stride: 2
    
    e_m = Conv2DLayerPool((3,3, 512), 1, 2, e_m, "pool_4", load_bool)

    feature_map_save(e_m)

    print("pool_4 completed")
    
    # Input tensor: (27,27, 1)
    # Output tensor: (25,25, 1024)
    # Kernel Dimensions: 3 X 3
    # Number of Kernels: 1024
    
    e_m = Conv2DLayer((3,3), 1024, 1, e_m, "conv_9", load_bool)

    feature_map_save(e_m)

    print("conv_9 completed")
    
    # Input tensor: (25,25, 1024)
    # Output tensor: (23,23, 1024)
    # Kernel Dimensions: 3 X 3 X 1024
    # Number of Kernels: 1024
    
    e_m = Conv2DLayer((3, 3, 1024), 1024, 1, e_m, "conv_10", load_bool)

    feature_map_save(e_m)

    print("conv_10 completed")

    # Input tensor: (23,23, 1024)
    # Output tensor: (46, 46, 512)
    # Kernel Dimensions: 2 X 2 X 1024
    # Number of Kernels: 512
    # Stride: 2
    
    e_m = UpConv((2,2, 1024), 512, 2, e_m, "up_conv_1", load_bool)

    feature_map_save(e_m)

    print("up_conv_1 completed")

    # Input tensor: (46, 46, 512) & (55, 55, 512)
    # Output tensor: (46,46, 1024)
    
    e_m = Concatenate([e_m, skip_connection_4], "concate_1", load_bool)

    feature_map_save(e_m)

    print("concate_1 completed")

    # Input tensor: (46, 46, 1024)
    # Output tensor: (44, 44, 512)
    # Kernel Dimensions: 3 X 3 X 1024
    # Number of Kernels: 512
    
    e_m = Conv2DLayer((3,3,1024), 512, 1, e_m, "conv_11", load_bool)

    feature_map_save(e_m)

    print("conv_11 completed")

    # Input tensor: (44, 44, 512)
    # Output tensor: (42, 42, 512)
    # Kernel Dimensions: 3 X 3 X 512
    # Number of Kernels: 512
    
    e_m = Conv2DLayer((3,3,512), 512, 1, e_m, "conv_12", load_bool)

    feature_map_save(e_m)

    print("conv_12 completed")

    #Concatenate a skip connection to produce an out_tensor with channel depth of 512
    # Input tensor: (42, 42, 512)
    # Output tensor: (84, 84, 256)
    # Kernel Dimensions: 2 X 2 X 512
    # Number of Kernels: 512
    
    e_m = UpConv((2,2,512), 256, 2, e_m, "up_conv_2", load_bool)

    feature_map_save(e_m)

    print("up_conv_2 completed")

    # Input tensor: (84, 84, 256) & (120, 120, 256)
    # Output tensor: (84,84, 512)
    
    e_m = Concatenate([e_m, skip_connection_3], "Concatenate_2", load_bool)

    print("Concatenate_2 Completed")

    # Input tensor: (84, 84, 512)
    # Output tensor: (82, 82, 256)
    # Kernel Dimensions: 3 X 3 X 512
    # Number of Kernels: 256
    
    e_m = Conv2DLayer((3,3,512), 256, 1, e_m, "conv_13", load_bool)

    feature_map_save(e_m)

    print("conv_13 completed")

    # Input tensor: (82, 82, 256)
    # Output tensor: (80, 80, 256)
    # Kernel Dimensions: 3 X 3 X 256
    # Number of Kernels: 256
    
    e_m = Conv2DLayer((3,3,256), 256, 1, e_m, "conv_14", load_bool)

    feature_map_save(e_m)

    print("conv_14 completed")

    # Concatenate a skip connection to create an out tensor of channel depth 256
    # Input tensor: (80, 80, 256)
    # Output tensor: (160, 160, 128)
    # Kernel Dimensions: 2 X 2 X 256
    # Number of Kernels: 128
    # Stride: 2
    
    
    e_m = UpConv((2,2, 256), 128, 2, e_m, "up_conv_3", load_bool)

    feature_map_save(e_m)

    print("up_conv_3 completed")

    # Input tensor: (160, 160, 128) & (249, 249, 128)
    # Output tensor: (160, 160, 256)
    
    e_m = Concatenate([e_m, skip_connection_2], "Concat_3", load_bool)

    print("Concat_3 completed")

    # Input tensor: (160, 160, 256)
    # Output tensor: (158, 158, 198)
    # Kernel Dimensions: 3 X 3 X 256
    # Number of Kernels: 198
    
    e_m = Conv2DLayer((3,3,256), 198, 1, e_m, "conv_15", load_bool)

    feature_map_save(e_m)

    print("conv_15 completed")

    # Input tensor: (158, 158, 198)
    # Output tensor: (156, 156, 128)
    # Kernel Dimensions: 3 X 3 X 198
    # Number of Kernels: 128
    
    e_m = Conv2DLayer((3,3,198), 128, 1, e_m, "conv_16", load_bool)

    feature_map_save(e_m)

    print("conv_16 completed")

    # Concatenate a skip connection to create an out tensor with channel depth 128
    # Input tensor: (156, 156, 128)
    # Output tensor: (312, 312, 64)
    # Kernel Dimensions: 3 X 3 X 128
    # Number of Kernels: 64
    # Stride: 2
    
    e_m = UpConv((2,2, 128), 64, 2, e_m, "up_conv_4", load_bool)

    feature_map_save(e_m)

    print("up_conv_4 completed", load_bool)

    # Input tensor: (312, 312, 64) & (508, 508, 64)
    # Output tensor: (312, 312, 128)
    
    e_m = Concatenate([e_m, skip_connection_1], "Concat_4", load_bool)

    print("Concat_4 completed")

    # Input tensor: (312, 312, 128)
    # Output tensor: (284, 284, 64)
    # Kernel Dimensions: 29 X 29 X 64
    # Number of Kernels: 64
    
    e_m = Conv2DLayer((29,29,128), 64, 1, e_m, "conv_17", load_bool)

    feature_map_save(e_m)

    print("conv_17 completed")

    # Input tensor: (284, 284, 64)
    # Output tensor: (258, 258, 64)
    # Kernel Dimensions: 27 X 27 X 64
    # Number of Kernels: 64
    
    e_m = Conv2DLayer((27,27,64), 64, 1, e_m, "conv_18", load_bool)

    feature_map_save(e_m)

    print("conv_18 completed")

    # Input tensor: (258, 258, 64)
    # Output tensor: (256, 256, 64)
    # Kernel Dimensions: 3 X 3 X 64
    # Number of Kernels: 64
    
    e_m = Conv2DLayer((3,3,64), 64, 1, e_m, "conv_19", load_bool)

    feature_map_save(e_m)

    print("conv_19 completed")

    # Input tensor: (256, 256, 64)
    # Output tensor: (512, 512, 64)
    # Kernel Dimensions: 3 X 3 X 64
    # Number of Kernels: 64
    # Stride: 2

    e_m = UpConv((2,2, 64), 64, 2, e_m, "up_conv_5", load_bool)

    feature_map_save(e_m)

    print("up_conv_5 completed")

    # Input tensor: (512, 512, 64)
    # Output tensor: (512, 512, 2)
    # Kernel Dimensions: 1 X 1 X 64
    # Number of Kernels: 2
    # Stride: 1
    
    e_m = Conv2DLayer((1,1,64),1, 1, e_m, "conv_20", load_bool)

    feature_map_save(e_m)

    print("conv_20 completed")

    return e_m

#I will now define the functions used for backpropation based on the layer being propagated:




def conv_back(curr_layer, ahead_layer, dC_dOn):
    # In order to calculate dC/dk(n-1) we must calculate multiple individual terms
    n_out = ahead_layer.out_gen()

    n_1_out = curr_layer.out_gen()

    n_2_out = curr_layer.input_source.out_gen()

    # First we will calculate dC/dO(n-1) = dOn/dRn * dRn/dO(n-1) * dC/dOn
    dOn_dRn = vectorize_derv_relu(n_out)

    k_rot = np.rot90(ahead_layer.kernels, k=2, axes=(1, 2))

    print(dOn_dRn.shape)

    print(np.multiply(dOn_dRn, dC_dOn).shape)

    print(k_rot.shape)

    print(n_1_out.shape)

    dc_dO_n_1 = []

    dC_dRn = np.multiply(dOn_dRn, dC_dOn)

    
    

    for index in range(0,ahead_layer.kernel_num):
        kernel = k_rot[index]
        out_dim = (len(dC_dRn) - 1) * 1 + len(kernel)
        up_img = np.zeros((out_dim, out_dim), dtype= np.float64)
        for i in range(0, len(dC_dRn)):
            for j in range(0, len(dC_dRn[i])):
                voxel = dC_dRn[i,j]
                patch = np.multiply(kernel, voxel[None,None, :])
                up_img[i : i + len(kernel), j : j + len(kernel)] += np.sum(patch, axis= -1)
        dc_dO_n_1.append(up_img)
    
    dc_dO_n_1 = np.transpose(np.stack(dc_dO_n_1), (1,2,0))

    print(dc_dO_n_1.shape)

    # Next we will determine the value of dR(n-1) / dk(n-1)

    dRn1_dkn1 = n_2_out

    # Finally we will calculate dO(n-1)/dR(n-1)

    dOn1_dRn1 = vectorize_derv_relu(n_1_out)

    #Final calculation

    error_signal = np.multiply(dOn1_dRn1, dc_dO_n_1)

    print(error_signal.shape)

    dC_dk1 = []

    for index in range(0, curr_layer.kernel_num):
        array_holder = []
        for pos in range(0,  curr_layer.input_source.kernel_num):
            array_holder.append(convolution(dRn1_dkn1[..., pos], error_signal[..., index], 1))
        array_holder = np.transpose(np.stack(array_holder), (1,2,0))
        dC_dk1.append(array_holder)
    
    
    dC_dk1 = np.stack(dC_dk1)
    print(dC_dk1.shape)
    np.save(f"back_prop_gradients\{curr_layer.name}.npy", dC_dk1)



# Function name is self-explanatory
def backpropagation(model, g_t):
    # Based on the model architecture, we know that the model will be at its last layer when inputted
    # g_t is the ground truth being used for backpropagation
    # Firts we will calculate the error signal via the output of the last layer (On) times 2(On - t) where t is the ground truth
    # Kernels can be automatically updated after gradient is caclulated because they aren't used in further backpropagation
    g_t_img = imread(g_t)

    gt_array = np.asarray(g_t_img)
    
    output_n = model.out_gen()

    output_n_1 = model.input_source.out_gen()

    error_signal = np.multiply(vectorize_derv_relu(output_n), 2*(output_n - gt_array))

    kernel_gradient = []

    for index in range(0, model.input_source.kernel_num):

        kernel_gradient.append(convolution(output_n_1[..., index], error_signal, 1))

    kernel_gradient = np.transpose(np.stack(kernel_gradient), (1,2,0))

    print(kernel_gradient.shape)

    while model.name != "conv_19":
        model = model.input_source

    conv_back(model.input_source, model, np.random.rand(256, 256, 64))
    # A correctness check measure




input_image = r"DIC-C2DH-HeLa\01\t014.tif"

#The cell tracking model will be an adaption of the UNet architecture with some modifications. 



image = imread(input_image)

image_array = np.asarray(image)


curr_model = encoder_block(image_array, True)

backpropagation_test = backpropagation(curr_model, "test.tif")
