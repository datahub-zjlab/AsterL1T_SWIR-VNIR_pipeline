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

def plot_aster(aster,select_bands=[0,1,2],scale=True):
    img = aster[select_bands].transpose(1,2,0)
    if scale:
        img = np.uint8(img/np.max(img)*255)
    f = plt.imshow(img)
    plt.show()
    return f