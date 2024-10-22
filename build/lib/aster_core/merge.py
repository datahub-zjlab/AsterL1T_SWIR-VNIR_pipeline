import numpy as np
from aster_core.cloud import get_cloud_masks,add_to_chanel,split_from_chanel
import copy

def merge_min(ref_list, ref_c=0):
    """
    合并多个数组，取每个位置的最小值。

    参数:
    ref_list (list of np.ndarray): 包含多个数组的列表，每个数组的形状可以是 (H, W) 或 (C, H, W)。

    返回:
    np.ndarray: 合并后的数组，形状为 (H, W) 或 (C, H, W)，每个位置的值为输入数组在该位置的最小值。
    """

    # 检查输入数组的形状
    shapes = [arr.shape for arr in ref_list]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError("All input arrays must have the same shape")

    # 获取第一个数组的形状
    shape = shapes[0]

    # 如果输入数组的形状是 (H, W)，则将其转换为 (1, H, W)
    if len(shape) == 2:
        ref_list = [arr[np.newaxis, :, :] for arr in ref_list]
        shape = (1,) + shape

    # 将列表中的数组沿第一个维度堆叠
    data = np.stack(ref_list, axis=0)

    # 提取第一个通道的数据
    data_bref = data[:, ref_c, :, :]

    # 将数据中的0值替换为无穷大，以便在计算最小值时忽略这些值
    data_bref[data_bref == 0] = np.inf

    # 计算每个位置的最小值对应的下标
    min_indices = np.argmin(data_bref, axis=0)

    # 根据最小值下标从原始数据中提取最小值
    min_values = np.squeeze(np.take_along_axis(data, np.expand_dims(np.expand_dims(min_indices, axis=0), axis=0), axis=0))

    # 如果输入数组的形状是 (H, W)，则将输出数组的形状转换为 (H, W)
    if len(shape) == 2:
        min_values = min_values[0, :, :]
        
    min_values[np.isinf(min_values)]=0
    min_values[np.isnan(min_values)]=0

    return min_values

def merge_mean(ref_list):
    """
    合并多个数组，取每个位置的平均值。

    参数:
    ref_list (list of np.ndarray): 包含多个数组的列表，每个数组的形状为 (B, C, H, W)。

    返回:
    np.ndarray: 合并后的数组，形状为 (C, H, W)，每个位置的值为输入数组在该位置的平均值。
    """
    # 将列表中的数组沿第一个维度堆叠
    data = np.stack(ref_list, axis=0)
    
    # 计算每个位置的平均值
    merge_data = np.mean(data, axis=0)
    
    merge_data[np.isinf(merge_data)]=0
    merge_data[np.isnan(merge_data)]=0
    
    return merge_data

def merge_deshadow(ref_list, ref_c=0, threshold=0.2):
    """
    合并多个数组，去除阴影。

    参数:
    ref_list (list of np.ndarray): 包含多个数组的列表，每个数组的形状为 (B, C, H, W)。
    threshold (float): 阈值，用于判断是否为阴影。

    返回:
    np.ndarray: 合并后的数组，形状为 (C, H, W)，每个位置的值为去除阴影后的值。
    """
    # 将列表中的数组沿第一个维度堆叠
    data = np.stack(ref_list, axis=0)
    
    # 提取第一个通道的数据
    data_bref = data[:,ref_c,:,:]
    
    # 将数据中的0值替换为无穷大，以便在计算最小值时忽略这些值
    data_bref[data_bref==0] = np.inf
    
    # 计算每个位置的最小值对应的下标
    min_indices = np.argmin(data_bref, axis=0)
    
    # 使用 np.argpartition 找到次小值的下标
    partitioned_indices = np.argpartition(data_bref, 1, axis=0)
    second_min_indices = partitioned_indices[1]
    
    data[np.isinf(data)]=0
    data[np.isnan(data)]=0
    
    # 根据最小值下标从原始数据中提取最小值
    min_values = np.squeeze(np.take_along_axis(data, np.expand_dims(np.expand_dims(min_indices, axis=0), axis=0), axis=0))
    
    # 根据次小值下标从原始数据中提取次小值
    second_min_values = np.squeeze(np.take_along_axis(data, np.expand_dims(np.expand_dims(second_min_indices, axis=0), axis=0), axis=0))
    
    # 计算权重
    cal_weights = 1 - (second_min_values[0]) / threshold
    
    # 初始化权重数组
    weights = np.ones_like(min_values[0])
    
    # 根据阈值调整权重
    weights[second_min_values[0] < threshold] = cal_weights[second_min_values[0] < threshold]
    
    # 根据权重合并最小值和次小值
    merge_data = min_values * weights + second_min_values * (1 - weights)
    
    merge_data[np.isinf(merge_data)]=0
    merge_data[np.isnan(merge_data)]=0
    
    return merge_data

import numpy as np

def sort_by_cloud_and_coverage(ref_list, mask_list, nodata=0):
    """
    Sorts images by cloud coverage and coverage percentage, and returns the top images based on criteria.

    Parameters:
    ref_list (list): List of reference images.
    mask_list (list): List of mask images corresponding to the reference images.

    Returns:
    tuple: A tuple containing two numpy arrays, the first with the selected reference images and the second with the corresponding mask images.
    """
    # Initialize lists to store percentages and intermediate values
    coverage_percentages = []
    p1_values = []
    p2_values = []

    # Iterate over each mask and corresponding image
    for index, mask in enumerate(mask_list):
        img = ref_list[index]
        
        # Calculate P1: Percentage of black pixels in the image
        P1 = (img[0] == nodata).sum() / np.prod(img[0].shape)
        
        # Calculate P2: Percentage of mask pixels over non-black pixels in the image
        P2 = mask.sum() / (img != 0).sum()
        
        # Calculate the combined percentage for sorting
        combined_percentage = 10 * (20 * P2 + P1)
        coverage_percentages.append(combined_percentage)
        p1_values.append(P1)
        p2_values.append(P2)

    # Combine data for sorting
    data_mask_combined = zip(ref_list, mask_list, coverage_percentages, p1_values, p2_values)
    
    # Sort the combined data based on the coverage percentage
    sorted_data_mask = sorted(data_mask_combined, key=lambda x: x[2])

    # Select images based on the sorted criteria
    if sorted_data_mask[0][2] <= 1 and sorted_data_mask[1][2] <= 5:
        selected_data = [img[0] for img in sorted_data_mask[:2]]
        selected_mask = [img[1] for img in sorted_data_mask[:2]]
        
    elif sorted_data_mask[0][2] <= 1 and sorted_data_mask[1][2] > 5:
        selected_data = [img[0] for img in sorted_data_mask[:1]]
        selected_mask = [img[1] for img in sorted_data_mask[:1]]

    else:
        selected_data = [img[0] for img in sorted_data_mask]
        selected_mask = [img[1] for img in sorted_data_mask]

    return selected_data, selected_mask

def get_nocloud_img(ref_list, mask_list):
    noholl_data = copy.deepcopy(ref_list)
    for index, data in enumerate(noholl_data):
        data[:, mask_list[index] == 1] = 0
        noholl_data[index] = data

    noholl_data=np.asarray(noholl_data)
    ref_list=np.asarray(ref_list)
    noholl_sum = noholl_data.sum((0, 1))
    zero_index = np.where(noholl_sum == 0)
    for i in range(len(zero_index[0])):
        noholl_data[:, :, zero_index[-2][i], zero_index[-1][i]] = ref_list[:, :, zero_index[-2][i], zero_index[-1][i]]
    return list(noholl_data)

def get_least_index(ref_list):
    ref_data = np.asarray(ref_list)
    if ref_data.shape[0] < 2:
        return 1
    
    index = np.full(ref_data.shape[2:], np.nan)

    i = 0
    while np.isnan(index).any():
        if i >= ref_data.shape[0]:
            break
        
        img = ref_data[i, 0]
        img[~np.isnan(index)] = 0
        pixel_x, pixel_y = np.where(img != 0)
        index[pixel_x, pixel_y] = int(i)
        i += 1
    
    return i

def get_least_ref_and_mask(ref_list, mask_list):
    """
    Get the index of the least value in the reflectance list and return the reflectance and mask lists up to that index.

    Parameters:
    ref_list (list): List of reflectance data
    mask_list (list): List of mask data

    Returns:
    tuple: A tuple containing two lists, the reflectance list and mask list up to the least value index.
    """
    nocloudref_list = get_nocloud_img(ref_list,mask_list)
    least_index = get_least_index(nocloudref_list)  # Get the index of the least value
    least_ref_list = ref_list[:least_index]  # Get the reflectance list up to the least value index
    least_mask_list = mask_list[:least_index]  # Get the mask list up to the least value index

    return least_ref_list, least_mask_list


def merge_deshadow_with_cloud_mask(ref_list, cloud_mask_list=None, ref_c=0, threshold=0.2, nodata=0, return_mask_in_chanel_flag=False):
    """
    Merge deshadowed reflectance data with cloud mask data.

    Parameters:
    ref_list (list): List of reflectance data
    cloud_mask_list (list, optional): List of cloud mask data, if not provided, it will be generated automatically
    ref_c (int, optional): Reference channel, default is 0
    threshold (float, optional): Threshold value, default is 0.2
    nodata (int, optional): No data value, default is 0
    return_mask_in_chanel_flag (bool, optional): Whether to return the mask in the channel, default is True

    Returns:
    np.ndarray or tuple: If return_mask_in_chanel_flag is True, returns the merged data; otherwise, returns the merged data and mask.
    """
    if cloud_mask_list is None:
        cloud_mask_list = get_cloud_masks(ref_list)  # Generate cloud masks if not provided

    sorted_ref_list, sorted_cloud_mask_list = sort_by_cloud_and_coverage(ref_list, cloud_mask_list, nodata=nodata)  # Sort by cloud coverage and coverage percentage

    least_ref_list, least_mask_list = get_least_ref_and_mask(sorted_ref_list, sorted_cloud_mask_list)  # Get the reflectance and mask lists up to the least value index
    ref_add_mask_list = add_to_chanel(least_ref_list, least_mask_list)  # Concatenate reflectance data with mask data

    if len(ref_add_mask_list) > 1:
        merge_data = merge_deshadow(ref_add_mask_list, ref_c=ref_c, threshold=threshold)  # Merge deshadowed data
    else:
        merge_data = ref_add_mask_list[0]  # If there is only one data, use it directly

    if return_mask_in_chanel_flag:
        return merge_data  # Return the merged data if mask is in the channel
    else:
        merge_data, merge_mask = split_from_chanel(merge_data)  # Otherwise, split the merged data and mask
        return merge_data, merge_mask  # Return the split data and mask

def merge_deshadow_with_cloud_mask_nosort(ref_list, cloud_mask_list=None, ref_c=0, threshold=0.2, nodata=0, return_mask_in_chanel_flag=False):
    """
    Merge deshadowed reflectance data with cloud mask data.

    Parameters:
    ref_list (list): List of reflectance data
    cloud_mask_list (list, optional): List of cloud mask data, if not provided, it will be generated automatically
    ref_c (int, optional): Reference channel, default is 0
    threshold (float, optional): Threshold value, default is 0.2
    nodata (int, optional): No data value, default is 0
    return_mask_in_chanel_flag (bool, optional): Whether to return the mask in the channel, default is True

    Returns:
    np.ndarray or tuple: If return_mask_in_chanel_flag is True, returns the merged data; otherwise, returns the merged data and mask.
    """
    if cloud_mask_list is None:
        cloud_mask_list = get_cloud_masks(ref_list)  # Generate cloud masks if not provided

    ref_add_mask_list = add_to_chanel(ref_list, cloud_mask_list)  # Concatenate reflectance data with mask data

    if len(ref_add_mask_list) > 1:
        merge_data = merge_deshadow(ref_add_mask_list, ref_c=ref_c, threshold=threshold)  # Merge deshadowed data
    else:
        merge_data = ref_add_mask_list[0]  # If there is only one data, use it directly

    if return_mask_in_chanel_flag:
        return merge_data  # Return the merged data if mask is in the channel
    else:
        merge_data, merge_mask = split_from_chanel(merge_data)  # Otherwise, split the merged data and mask
        return merge_data, merge_mask  # Return the split data and mask


