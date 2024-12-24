import numpy as np
# from aster_core.cloud import get_cloud_masks,add_to_chanel,split_from_chanel
from skimage import filters
from aster_core.color_transfer import colorFunction
from aster_core.cloud import cal_cloud_mask,cal_spectral_info
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse



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

'''
MERGE_CUSTOM
'''

class AST_TILE():
    def __init__(self, reflectance, cloud_cover, aod, solar_z, solar_a, atmos_profile, nodata_value=0, toa=None):
        
        self.reflectance = reflectance
        self.cloud_cover = cloud_cover
        self.aod = aod
        self.solar_z = solar_z
        self.solar_a = solar_a
        self.atmos_profile = atmos_profile

        self.nodata_value = nodata_value
        self.coverage = self.cal_coverage()
    
    def cal_coverage(self,band_id=0):
        coverage = calculate_coverage(self.reflectance,band_id=band_id,nodata_value=self.nodata_value)
        return coverage
    
    # def cal_fg(self):
    #     self.fg = common_used_functional_group(self.reflectance)

    def cal_mask(self,img):
        mask = img != self.nodata_value
        return mask

    def cal_cloud_score(self,no_cloud_reflectance):

        mask_0 = self.cal_mask(self.reflectance[0])
        mask_1 = self.cal_mask(no_cloud_reflectance[0])

        overlap_area = np.logical_and(mask_0,mask_1)

        spectral_0 = cal_spectral_info(no_cloud_reflectance,mask=overlap_area,percent=(70,100))
        spectral_1 = cal_spectral_info(self.reflectance,mask=overlap_area,percent=(70,100))
        
        cloud_score_0 = np.corrcoef(spectral_0,spectral_1)[0,1]
        
        cloud_score_1 = ssim(no_cloud_reflectance[0],self.reflectance[0],data_range=1)/(np.sum(overlap_area)/np.sum(mask_1))
  

        self.cloud_score = cloud_score_0*0.5+cloud_score_1*0.5

        return self.cloud_score

def calculate_coverage(image,band_id=0,nodata_value=0):
    """
    Calculate the effective pixel coverage of the image based on band 1.
    :param image: Atmospherically corrected image, shape (9, 1024, 1024)
    :return: Coverage percentage, range 0 to 100
    """
    band1 = image[band_id, :, :]  # Get band 1
    valid_pixels = np.sum(band1 != nodata_value)  # Assume 0 is invalid pixel
    total_pixels = 1024 * 1024
    coverage = (valid_pixels / total_pixels) * 100
    return coverage

def is_overlap_nodate(img1,img2,nodata_value = 0):
    img1_band1 = img1[0]
    img2_band1 = img2[0]
    img1_band1_missing_mask = img1_band1 == nodata_value
    img2_band1_valid_mask = img2_band1 != nodata_value
    if np.any(img1_band1_missing_mask*img2_band1_valid_mask):
        return True
    else:
        return False

def correct_and_fill_images(img1, img2,nodata_value=0):
    """
    Correct img2 based on the overlap with img1, then fill the missing regions of img1 with data from img2.
    :param img1: First candidate image
    :param img2: Second candidate image
    :return: Filled first candidate image
    """
    # Implement correction and filling logic
    corrected_img2 = colorFunction(img1,img2)  # Placeholder, return original image
    filled_img1 = img1  # Placeholder, return original image
    filled_img1[:,filled_img1[0]==nodata_value] = corrected_img2[:,filled_img1[0]==nodata_value]
    return filled_img1

def merge_custom(ref_list, min_num=1000, lower_percent=99.99):
    no_cloud_min_aster = merge_min(ref_list)
    thresh_otsu = filters.threshold_otsu(no_cloud_min_aster[0],nbins=100)
    invalid_mask = no_cloud_min_aster[0]>thresh_otsu

    cf_img_list = []

    for img in ref_list:
        try:
            img[img<0]=0

            rough_cloud_mask = cal_cloud_mask(img,lower_percent=lower_percent)
            cf_mask = invalid_mask & (img[0]>0) & ~rough_cloud_mask

            if np.sum(cf_mask)<min_num:
                continue

            cf_img = colorFunction(no_cloud_min_aster,img,mask=cf_mask)
            cf_img[:,rough_cloud_mask] = img[:,rough_cloud_mask]

            accurate_cloud_mask = rough_cloud_mask

            cf_img[:,accurate_cloud_mask] = np.nan
            cf_img[:,cf_img[0]==0] = np.nan

            if not np.any(np.iscomplex(cf_img)):
                cf_img_list.append(cf_img)
        except:
            continue
        
    cf_img_list.append(no_cloud_min_aster)
    cf_img_list.append(no_cloud_min_aster)
    first_reference_aster = np.nanmedian(cf_img_list,axis=0)

    return first_reference_aster

def merge_region(ref_list, min_num=1000, alpha=0.5,beta=0.9, mse_threshold=0.5):

    def calculate_coverage(ref):
        """Calculate the coverage of a single matrix"""
        sum_ref = np.sum(ref, axis=0)  # Sum across all bands
        nonzero_count = np.count_nonzero(sum_ref)  # Count the number of non-zero pixels
        coverage = nonzero_count / (1024 * 1024)  # Calculate the coverage
        return coverage

    def sort_ref_list_by_coverage(ref_list):
        """Sort ref_list by coverage in descending order"""
        # Calculate the coverage for each matrix
        coverages = [calculate_coverage(ref) for ref in ref_list]
        
        # Pair each coverage with its corresponding matrix
        coverage_ref_pairs = list(zip(coverages, ref_list))
        
        # Sort the pairs by coverage in descending order
        coverage_ref_pairs.sort(reverse=True, key=lambda x: x[0])
        
        # Extract the sorted matrices
        sorted_ref_list = [pair[1] for pair in coverage_ref_pairs]
        
        return sorted_ref_list
    
    custom_result = merge_custom(ref_list)
    first_reference_img = np.zeros_like(custom_result)
    sorted_ref_list = sort_ref_list_by_coverage(ref_list)

    for img in sorted_ref_list:
        try:
            img[img<0]=0.0001

            rough_cloud_mask = cal_cloud_mask(img,lower_percent=99.99)
            cf_mask = (img[0]>0) & ~rough_cloud_mask

            if np.sum(cf_mask)<min_num:
                continue
        
            cf_img = colorFunction(custom_result,img,mask=cf_mask,alpha=alpha,beta=beta)
            cf_img[:,rough_cloud_mask] = img[:,rough_cloud_mask]

            # cf_max_img = colorFunction(custom_result,img,mask=cf_mask,alpha=1,beta=1)
            # cf_max_img[:,rough_cloud_mask] = img[:,rough_cloud_mask]

            if not np.any(np.iscomplex(cf_img)):

                empty_region = (np.sum(first_reference_img,axis=0)==0) & (np.sum(cf_img,axis=0)!=0)
                covered_region = (np.sum(first_reference_img,axis=0)!=0) & (np.sum(cf_img,axis=0)!=0)

                if np.any(empty_region):
                    
                    first_reference_img[:,empty_region] = cf_img[:,empty_region]

                if np.any(covered_region):

                    orig_coverd_region_mse = mse(first_reference_img[:,covered_region],custom_result[:,covered_region])
                    target_coverd_region_mse = mse(cf_img[:,covered_region],custom_result[:,covered_region])

                    if (target_coverd_region_mse<orig_coverd_region_mse) & ((orig_coverd_region_mse-target_coverd_region_mse)/orig_coverd_region_mse>mse_threshold):
                        first_reference_img[:,covered_region] = cf_img[:,covered_region]

        except:
            continue
    return first_reference_img

def merge_region_with_identifier_channel(ref_list, min_num=1000, alpha=0.5, beta=0.9, mse_threshold=0.5):

    def calculate_coverage(ref):
        """Calculate the coverage of a single matrix"""
        sum_ref = np.sum(ref, axis=0)  # Sum across all bands
        nonzero_count = np.count_nonzero(sum_ref)  # Count the number of non-zero pixels
        coverage = nonzero_count / (sum_ref.shape[0] * sum_ref.shape[1])  # Calculate the coverage
        return coverage

    def sort_ref_list_by_coverage(ref_list):
        """Sort ref_list by coverage in descending order"""
        # Calculate the coverage for each matrix
        coverages = [calculate_coverage(ref) for ref in ref_list]
        
        # Pair each coverage with its corresponding matrix and original index
        coverage_ref_pairs = list(zip(coverages, range(len(ref_list)), ref_list))
        
        # Sort the pairs by coverage in descending order
        coverage_ref_pairs.sort(reverse=True, key=lambda x: x[0])
        
        # Extract the sorted matrices and their original indices
        sorted_ref_list = [pair[2] for pair in coverage_ref_pairs]
        original_indices = [pair[1] for pair in coverage_ref_pairs]
        
        return sorted_ref_list, original_indices
    
    custom_result = merge_custom(ref_list, min_num=min_num)
    first_reference_img = np.zeros_like(custom_result)

    _,x,y = first_reference_img.shape
    
    # Sort ref_list by coverage and get the original indices
    sorted_ref_list, original_indices = sort_ref_list_by_coverage(ref_list)

    # Initialize the identifier channel
    identifier_channel = np.zeros((x, y), dtype=int)  # New channel for identifiers

    for idx, img in zip(original_indices, sorted_ref_list):
        try:
            img[img < 0] = 0.0001  # Set negative values to a small positive value

            rough_cloud_mask = cal_cloud_mask(img, lower_percent=99.99)
            cf_mask = (img[0] > 0) & ~rough_cloud_mask

            if np.sum(cf_mask) < min_num:
                continue
        
            cf_img = colorFunction(custom_result, img, mask=cf_mask, alpha=alpha, beta=beta)
            cf_img[:, rough_cloud_mask] = img[:, rough_cloud_mask]

            # Add a new channel to cf_img with the identifier value
            identifier_value = idx + 1  # Unique identifier for this ref (starting from 1)
            identifier_img = np.full((x, y), identifier_value, dtype=int)

            if not np.any(np.iscomplex(cf_img)):

                empty_region = (np.sum(first_reference_img, axis=0) == 0) & (np.sum(cf_img, axis=0) != 0)
                covered_region = (np.sum(first_reference_img, axis=0) != 0) & (np.sum(cf_img, axis=0) != 0)

                if np.any(empty_region):
                    # Update first_reference_img with cf_img in empty regions
                    first_reference_img[:, empty_region] = cf_img[:, empty_region]
                    # Update identifier_channel with the identifier value in empty regions
                    identifier_channel[empty_region] = identifier_img[empty_region]

                if np.any(covered_region):
                    orig_covered_region_mse = mse(first_reference_img[:, covered_region], custom_result[:, covered_region])
                    target_covered_region_mse = mse(cf_img[:, covered_region], custom_result[:, covered_region])

                    if (target_covered_region_mse < orig_covered_region_mse) & ((orig_covered_region_mse - target_covered_region_mse) / orig_covered_region_mse > mse_threshold):
                        # Update first_reference_img with cf_img in covered regions
                        first_reference_img[:, covered_region] = cf_img[:, covered_region]
                        # Update identifier_channel with the identifier value in covered regions
                        identifier_channel[covered_region] = identifier_img[covered_region]

        except:
            continue

    return first_reference_img, identifier_channel


