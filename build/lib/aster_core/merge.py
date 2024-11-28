import numpy as np
# from aster_core.cloud import get_cloud_masks,add_to_chanel,split_from_chanel
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



