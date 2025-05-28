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

# idk how stride would work for up_conv

def up_conv (img_array, kernel):
    up_img = [[0 for i in range(0, (len(kernel) * len(img_array)))] for j in range(0, (len(kernel) * len(img_array)))]
    up_img = np.asarray(up_img)
    for i in range(0, len(img_array)):
        for j in range(0, len(img_array)):
            up_img[(len(kernel) * i) : ((len(kernel) * i) + len(kernel)), (len(kernel) * j) : ((len(kernel) * j) + len(kernel))] = img_array[i,j] * kernel
    return up_img



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
            feature_map = convolution(input_data, kernel, self.c_stride)
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
    
class Conv2DLayer(convBaseLayer):
    def get_input(self):
        return self.input_source.out_gen()
    
    def transform_output(self, feature_map):
        return vectorize_reLu(feature_map)
    
    def prev_dmg_f_maps(self):
        return os.path.exists(f"{self.input_source.name}_feature_maps")
    
class Conv2DLayerPool(convBaseLayer):
    def get_input(self):
        return self.input_source.out_gen()
    
    def transform_output(self, feature_map):
        return normalize_array(vectorize_reLu(feature_map), 0, 1)
    
    def prev_dmg_f_maps(self):
        return os.path.exists(f"{self.input_source.name}_feature_maps")


def feature_map_save(layer):
    if not os.path.exists(f"{layer.name}_feature_maps"):
        os.makedirs(f"{layer.name}_feature_maps")

    output_tensor = layer.out_gen()
    for index in range(0, output_tensor.shape[2]):
        imwrite(f"{layer.name}_feature_maps\out_feature_map_{index}.tif", output_tensor[:,:,index])




def encoder_block(input_img):
    e_m = InputLayer((3,3), 64, 1, input_img, "conv_1")

    feature_map_save(e_m)

    print("conv_1 completed")

    e_m = Conv2DLayer((3,3, 64), 64, 1, e_m, "conv_2")

    feature_map_save(e_m)

    print("conv_2 completed")

    e_m = Conv2DLayerPool((3,3, 64),1,2,e_m, "pool_1")

    feature_map_save(e_m)

    print("pool_1 completed")
    
    e_m = Conv2DLayer((3,3), 128, 1, e_m, "conv_3")

    feature_map_save(e_m)

    print("conv_3 completed")
    
    e_m = Conv2DLayer((3,3, 128), 128, 1, e_m, "conv_4")

    feature_map_save(e_m)

    print("conv_4 completed")
    
    e_m = Conv2DLayerPool((3,3, 128),1,2,e_m,"pool_2")

    feature_map_save(e_m)

    print("pool_2 completed")
    
    e_m = Conv2DLayer((3,3), 256, 1, e_m ,"conv_5")

    feature_map_save(e_m)

    print("conv_5 completed")
    
    e_m = Conv2DLayer((3,3, 256),256, 1, e_m, "conv_6")

    feature_map_save(e_m)

    print("conv_6 completed")
    
    e_m = Conv2DLayerPool((3,3, 256),1,2,e_m,"pool_3")

    feature_map_save(e_m)

    print("pool_3 completed")
    
    e_m = Conv2DLayer((3,3),512, 1, e_m, "conv_7")

    feature_map_save(e_m)

    print("conv_7 completed")
    
    e_m = Conv2DLayer((3,3, 512), 512, 1, e_m, "conv_8")

    feature_map_save(e_m)

    print("conv_8 completed")
    
    e_m = Conv2DLayerPool((3,3, 512), 1, 2, e_m, "pool_4")

    feature_map_save(e_m)

    print("pool_4 completed")
    
    e_m = Conv2DLayer((3,3), 1024, 1, e_m, "conv_9")

    feature_map_save(e_m)

    print("conv_9 completed")
    
    e_m = Conv2DLayer((3, 3, 1024), 1024, 1, e_m, "conv_10")

    feature_map_save(e_m)

    print("conv_10 completed")



input_image = r"DIC-C2DH-HeLa\01\t014.tif"

#The cell tracking model will be an adaption of the UNet architecture with some modifications. 



image = imread(input_image)

image_array = np.asarray(image)


curr_model = encoder_block(image_array)
