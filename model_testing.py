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


def feature_map_save(tensor, pos, layer, row, col):
    if not os.path.exists(f"layer_{pos}_{layer.name}_feature_maps"):
        os.makedirs(f"layer_{pos}_{layer.name}_feature_maps")

    output_tensor = tensor
    for index in range(0, output_tensor.shape[-1]):
        imwrite(f"layer_{pos}_{layer.name}_feature_maps\out_feature_map_{index}_row_{row}_col_{col}.tif", output_tensor[:,:,:,index])

cell_seg_model = keras.models.load_model("image_segmentation.keras")

print("model loaded")

test_input = imread(r'DIC-C2DH-HeLa\01\t032.tif')

expected_output = imread(r'processed_data\cell_distance\man_seg032.tif')

print(np.max(expected_output))

test_input = np.expand_dims(test_input, axis=(0, -1))

print(test_input.shape)

prediction  = cell_seg_model.predict(test_input[:, :256, : 256])

print(np.max(prediction))

imwrite("predicted_model_seg.tif", cell_seg_model.predict(test_input[:, :256, : 256]))

"""for row in range(1,3):
    for col in range(1,3):
        for i,layer in enumerate(cell_seg_model.layers):
            intermediate_model = keras.Model(inputs=cell_seg_model.input, outputs = layer.output)
            test_output = intermediate_model.predict(test_input[:, (row - 1) * 256 : row * 256, (col - 1) * 256 : col * 256])
            print(test_output.shape)
            feature_map_save(test_output, i, layer, row, col)"""