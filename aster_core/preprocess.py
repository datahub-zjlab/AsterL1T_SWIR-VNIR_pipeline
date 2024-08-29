import numpy as np
import cv2

from aster_core.hdf_utils import parse_meta,get_irradiance1,get_ucc1,dn2rad,rad2ref

def set_boundary_inner_pixels_to_nodata(data, nodata_value=0, erosion_kernel_size=16):
    # data -> (9, 1024, 1024)
    
    # 初始化一个与数据形状相同的掩码，全部设置为1
    mask = np.ones_like(data[0, :, :], dtype=np.uint8)
    
    # 遍历每个通道，生成每个通道的掩码并进行逻辑与操作
    for i in range(data.shape[0]):
        channel_mask = (data[i, :, :] != nodata_value).astype(np.uint8)
        mask *= channel_mask
    
    # 对掩码进行腐蚀操作
    kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    
    # 取反掩码，以便将边界和内部像素区分开
    mask = ~eroded_mask.astype(bool)
    
    # 将边界和内部像素设置为nodata_value
    for i in range(data.shape[0]):
        data[i, mask] = nodata_value
    
    return data

def cal_radiance(aster_dn, meta, bands, nodata_value=0):
    meta_parser = parse_meta(meta)
    # Initialize lists to store radiance and toa arrays
    radiance_list = []
    for band_desc, sub_aster in zip(bands, aster_dn):
        # Get the unit conversion coefficient for the band
        ucc1 = get_ucc1(meta_parser, band_desc)
        # Convert digital numbers (DN) to radiance
        radiance = dn2rad(sub_aster, ucc1)
        # Set radiance values of 0 to None
        radiance[radiance == dn2rad(0, ucc1)] = nodata_value
        if np.count_nonzero(radiance) == 0:
            return None
        else:
            radiance_list.append(radiance)
    
    radiance = np.stack(radiance_list, axis=0)
    radiance = set_boundary_inner_pixels_to_nodata(radiance)
    radiance[np.isnan(radiance)]=nodata_value
    radiance[np.isinf(radiance)]=nodata_value

    return radiance

def cal_toa(aster_radiance,meta,bands,nodata_value=0):
    meta_parser = parse_meta(meta)
    # Initialize lists to store radiance and toa arrays
    toa_list = []
    for band_desc, radiance in zip(bands, aster_radiance):
        # Get the solar irradiance for the band
        irradiance1 = get_irradiance1(meta_parser, band_desc)
        toa = rad2ref(radiance, meta_parser['esd'], irradiance1, meta_parser['sza'])
        if np.count_nonzero(toa) == 0:
            return None
        else:
            toa_list.append(toa)

    toa = np.stack(toa_list, axis=0)
    toa[np.isnan(toa)]=nodata_value
    toa[np.isinf(toa)]=nodata_value
    
    return toa
