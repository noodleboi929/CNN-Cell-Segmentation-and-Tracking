import numpy as np
from tifffile import imread, imwrite
import cv2
import math
import os
from pathlib import Path


folder_path = "biases"
for file in os.listdir(folder_path):
    os.remove(f"biases\{file}")