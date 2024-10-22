import rasterio
from pyproj import CRS

import matplotlib.pyplot as plt
import numpy as np

def writeGeoTiff(output_file,data,geotransform,crs=CRS.from_epsg(3857),dtype=rasterio.float32):
    out_meta = {}
    out_meta.update({
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "transform": geotransform,
        "dtype":dtype,
        "crs":crs,
        "count":data.shape[0]
    })
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(data)

def plot_aster(aster,select_bands=[0,1,2],scale=True,max_value=None):
    img = aster[select_bands].transpose(1,2,0)
    if scale:
        if max_value is None:
            max_value = np.max(img)
        img = np.uint8(img/max_value*255)
    f = plt.imshow(img)
    plt.show()
    return f

def plot_aster_scale_per_chanel(aster, select_bands=[0, 1, 2], scale=True, max_value=None):
    # 选择指定的波段
    img = aster[select_bands].transpose(1, 2, 0)
    
    if scale:
        if max_value is None:
            # 计算每个波段的98%分位数
            max_values = np.percentile(img, 100, axis=(0, 1))
            min_values = np.percentile(img, 5, axis=(0, 1))
            img = (img - min_values) / max_values
        else:
            # 使用指定的最大值进行归一化
            img = img / max_value
        
        # 将归一化后的图像转换为8位无符号整数
        img = np.uint8(img * 255)
    
    # 显示图像
    f = plt.imshow(img)
    plt.show()
    return f

def plot_aster_sb(aster,scale=True,max_value=None):
    img = aster
    if scale:
        if max_value is None:
            max_value = np.max(img)
        img = np.uint8(img/max_value*255)
    f = plt.imshow(img)
    plt.show()
    return f