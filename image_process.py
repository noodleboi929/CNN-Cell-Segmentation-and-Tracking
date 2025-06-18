import numpy as np
from tifffile import imread, imwrite
import cv2
import os

image = imread(r'DIC-C2DH-HeLa\01_ST\SEG\man_seg032.tif')

def image_proces(path, name):
    image = imread(path)
    colors_tracker = []
    cells = []
    background = []
    for i in range(0,image.shape[0]):
        for j in range(0, len(image[i])):
            if image[i][j] not in colors_tracker and image[i][j] != 0:
                cells.append([(i,j)])
                colors_tracker.append(image[i][j])
                print(f"cords: {i} and {j}")
            elif image[i][j] in colors_tracker and image[i][j] != 0:
                cells[colors_tracker.index(image[i][j])].append((i,j))
            else:
                background.append((i,j))




    # euclidean disttance transform for cell distances
    e_u_d_tiff = [[0 for i in range(0, image.shape[1])] for j in range(0, image.shape[0])]
    e_u_d_tiff = np.asarray(e_u_d_tiff)
    scratch_array = np.zeros_like(image, dtype= np.uint8)  
    for cell in cells:
        cell_cords = np.array(list(cell))
        scratch_array[cell_cords[:, 0], cell_cords[:, 1]] = 255
        binary_mask = np.asarray(scratch_array, dtype=np.uint8)
        binary_mask[binary_mask > 0] = 255
        dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        dist_output = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
        #print("loop completed")
        e_u_d_tiff = np.add(e_u_d_tiff, dist)
                
            
    # euclidean distance transofrm for cell neighbor value
    n_scratch_array = np.zeros_like(image, dtype= np.uint8)
    agreggate_array = np.zeros_like(image, dtype= np.uint8)
    counter = 0
    for cell in cells:
        counter = counter + 1
        n_scratch_array = np.zeros_like(image, dtype= np.uint8)
        #Conversion protocol
        cell_cords = np.array(list(cell))
        bg_cords = np.array(list(background))
        n_scratch_array[cell_cords[:, 0], cell_cords[:, 1]] = 255
        n_scratch_array[bg_cords[:, 0], bg_cords[:, 1]] = 255
        #imwrite(f'invesion_{counter}.tif', n_scratch_array)
        #Distance Transform
        n_dist = cv2.distanceTransform(n_scratch_array, cv2.DIST_L2, 5)
        n_dist_output = cv2.normalize(n_dist, None, 0, 1.0, cv2.NORM_MINMAX)
        #imwrite(f'distance_transform_{counter}.tif', n_dist_output)

        scaled_output = (n_dist_output * 255).astype(np.uint8)

        #Cutting and Inversion
        for (x,y) in cell:
            #print(scaled_output[x][y])
            agreggate_array[x][y] = abs(scaled_output[x][y] - 255)
        #print("Cut and Inversion Completed")
        #imwrite(f'cut_and_inversion_{counter}.tif',  agreggate_array)

    #imwrite('n_dist_debug.tif', agreggate_array)
    #grayscale closing
    kernel = np.ones((5,5),np.uint8)
    # agreggate_array = agreggate_array.astype(np.uint8)
    closed_aggregate_array = cv2.morphologyEx(agreggate_array, cv2.MORPH_CLOSE, kernel)
    #closed_aggregate_array = cv2.normalize(closed_aggregate_array, None, 0, 1.0, cv2.NORM_MINMAX)
    #imwrite('n_dist_debug_2.tif', agreggate_array)
    #imwrite('n_dist_debug_3.tif', closed_aggregate_array)
    scaled_array = closed_aggregate_array ** 3
            
    print("Array scaled")
        
    imwrite(f'processed_data/cell_distance/{name}', e_u_d_tiff)
    imwrite(f'processed_data/neighbor_distance/{name}', scaled_array)

for file in os.listdir(r'DIC-C2DH-HeLa\01_ST\SEG'):
    full_path = os.path.join(r'DIC-C2DH-HeLa\01_ST\SEG', file)
    image_proces(full_path, file)
    print(f"{file} was processed")


