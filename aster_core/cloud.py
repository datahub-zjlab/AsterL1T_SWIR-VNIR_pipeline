import numpy as np
import cv2

def get_cloud_mask(aster_reflectance,threashold=[1,1/3,1/2,3/4,30,30,3,3]):
    '''
    aster_reflectance: [band,x,y]
    '''
    _,dx,dy = np.shape(aster_reflectance)

    B_G = aster_reflectance[0]
    B_R = aster_reflectance[1]
    B_NIR = aster_reflectance[2]
    B_SWIR_1 = aster_reflectance[3]
    B_SWIR_2 = aster_reflectance[5]

    # 添加一个小的常数来避免除零错误
    epsilon = 1e-10
    CI_11 = (B_NIR + B_SWIR_1) / (B_G + B_R + epsilon)
    CI_11[~np.isfinite(CI_11)] = 0
    CI_21 = (B_G + B_R + B_NIR + B_SWIR_1 + B_SWIR_2) / 5
    CSI = (B_NIR + B_SWIR_1) / 2

    T2 = np.mean(CI_21[CI_21 > 0]) + (np.max(CI_21[CI_21 > 0]) - np.mean(CI_21[CI_21 > 0])) * threashold[1]
    T3 = np.min(CSI[CSI > 0]) + (np.mean(CSI[CSI > 0]) - np.min(CSI[CSI > 0])) * threashold[2]
    T4 = np.min(B_G[B_G > 0]) + (np.mean(B_G[B_G > 0]) - np.min(B_G[B_G > 0])) * threashold[3]

    cloud = np.zeros((dx,dy))
    cloud[(abs(CI_11 - 1) < threashold[0]) & (CI_21 > T2)] = 1
    cloud[B_G == 0] = 0
    if cloud.dtype != np.uint8:
        cloud = cv2.convertScaleAbs(cloud)
    cloud_blur = cv2.medianBlur(cloud,threashold[6])
    kernel = np.ones((5, 5), np.uint8)
    cloud_blur = cv2.dilate(cloud_blur, kernel, 1)

    cloud_shadows = np.zeros((dx,dy))
    cloud_shadows[((CSI < T3) & (B_G < T4))] = 1
    cloud_shadows_pixel_x, cloud_shadows_pixel_y = np.where(cloud_shadows != 0)
    cloud_shadows_pixel = list(zip(cloud_shadows_pixel_x, cloud_shadows_pixel_y))
    for x, y in cloud_shadows_pixel:
        minx, maxx = max(0, x - threashold[4] // 2), min(dx, x + threashold[4] // 2)
        miny, maxy = max(0, y - threashold[5] // 2), min(dy, y + threashold[5] // 2)
        if not cloud_blur[minx:maxx, miny:maxy].sum():
            cloud_shadows[x, y] = 0
    cloud_shadows[B_G == 0] = 0
    if cloud_shadows.dtype != np.uint8:
        cloud_shadows = cv2.convertScaleAbs(cloud_shadows)
    cloud_shadows_blur = cv2.medianBlur(cloud_shadows, threashold[7])
    kernel = np.ones((5, 5), np.uint8)
    cloud_shadows_blur = cv2.dilate(cloud_shadows_blur, kernel, 1)

    cloud_blur[cloud_shadows_blur==1]=1

    return cloud_blur

def get_cloud_masks(aster_reflectance_list,threashold=[1,1/3,1/2,3/4,30,30,3,3]):
    cloud_mask_list = []
    for aster_reflectance in aster_reflectance_list:
        cloud_mask_list.append(get_cloud_mask(aster_reflectance,threashold=threashold))
    return cloud_mask_list

def add_to_chanel(aster_reflectance_list, cloud_mask_list):
    '''
    aster_reflectance: [band,x,y]
    cloud_mask: [x,y] 
    '''
    # 检查输入是矩阵还是列表
    if isinstance(aster_reflectance_list, np.ndarray) and isinstance(cloud_mask_list, np.ndarray):
        # 如果输入是矩阵，则将其视为单个元素进行处理
        output = np.concatenate((aster_reflectance_list, cloud_mask_list[np.newaxis, ...]), axis=0)
        return output
    elif isinstance(aster_reflectance_list, list) and isinstance(cloud_mask_list, list):
        # 如果输入是列表，则按列表处理
        if len(aster_reflectance_list) == 0 or len(cloud_mask_list) == 0 or len(cloud_mask_list) != len(aster_reflectance_list):
            raise ValueError("Input lists must have same length")
        
        output_list = []

        for aster_reflectance, cloud_mask in zip(aster_reflectance_list, cloud_mask_list):
            output = np.concatenate((aster_reflectance, cloud_mask[np.newaxis, ...]), axis=0)
            output_list.append(output)
        
        return output_list
    else:
        raise ValueError("Input must be either both matrices or both lists")

def split_from_chanel(input_list, mask_index=-1):
    '''
    input: [band+1,x,y]
    '''
    # 检查输入是矩阵还是列表
    if isinstance(input_list, np.ndarray):
        # 如果输入是矩阵，则将其视为单个元素进行处理
        aster_reflectance = input_list[:mask_index]
        mask = input_list[mask_index]
        return aster_reflectance, mask
    elif isinstance(input_list, list):
        # 如果输入是列表，则按列表处理
        aster_reflectance_list = []
        cloud_mask_list = []
        for input in input_list:
            aster_reflectance = input[:mask_index]
            aster_reflectance_list.append(aster_reflectance)
            mask = input[mask_index]
            cloud_mask_list.append(mask)
        return aster_reflectance_list, cloud_mask_list
    else:
        raise ValueError("Input must be either a matrix or a list")

