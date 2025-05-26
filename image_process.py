import numpy as np
from tifffile import imread, imwrite
import cv2

image = imread(r'DIC-C2DH-HeLa\01_ST\SEG\man_seg032.tif')
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



"""
# euclidean disttance transform for cell distances
e_u_d_tiff = [[0 for i in range(0, image.shape[1])] for j in range(0, image.shape[0])]
e_u_d_tiff = np.asarray(e_u_d_tiff)
scratch_array = [[0 for i in range(0, image.shape[1])] for j in range(0, image.shape[0])]  
for cell in cells:
    for i in range(0,image.shape[0]):
        for j in range(0,len(image[i])):
            if (i,j) in cell:
                scratch_array[i][j] = image[i][j]
            else:
                scratch_array[i][j] = 0
    binary_mask = np.asarray(scratch_array, dtype=np.uint8)
    binary_mask[binary_mask > 0] = 255
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    dist_output = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    print("loop completed")
    e_u_d_tiff = np.add(e_u_d_tiff, dist)
            
"""            
# euclidean distance transofrm for cell neighbor value
n_scratch_array = [[0 for i in range(0, image.shape[1])] for j in range(0, image.shape[0])]
agreggate_array = [[0 for i in range(0, image.shape[1])] for j in range(0, image.shape[0])]
agreggate_array = np.asarray(agreggate_array)
for cell in cells:
    #Conversion protocol
    for i in range(0,image.shape[0]):
        for j in range(0,len(image[i])):
            if (i,j) in cell:
                n_scratch_array[i][j] = 255
            elif (i,j) in background:
                n_scratch_array[i][j] = 255
            else:
                n_scratch_array[i][j] = 0
    #Distance Transform
    n_scratch_array = np.asarray(n_scratch_array, dtype= np.uint8)
    n_dist = cv2.distanceTransform(n_scratch_array, cv2.DIST_L2, 5)
    n_dist_output = cv2.normalize(n_dist, None, 0, 1.0, cv2.NORM_MINMAX)

    #Cutting and Inversion
    for (x,y) in cell:
        agreggate_array[x][y] = abs(n_dist_output[x][y] - 1.0)
    print("Cut and Inversion Completed")

imwrite('n_dist_debug.tif', agreggate_array)
#grayscale closing
kernel = np.ones((5,5),np.uint8)
agreggate_array = agreggate_array.astype(np.uint8)
closed_aggregate_array = cv2.morphologyEx(agreggate_array, cv2.MORPH_CLOSE, kernel)
imwrite('n_dist_debug_2.tif', agreggate_array)
imwrite('n_dist_debug_3.tif', closed_aggregate_array)
scaled_array = closed_aggregate_array ** 3
        
print("Array scaled")
    
#imwrite('test.tif', e_u_d_tiff)
imwrite('n_dist.tif', scaled_array)


