import numpy as np
from rasterio.coords import BoundingBox
from rasterio.warp import transform_bounds
from shapely.geometry import box,Polygon
from rasterio.transform import Affine

'''
Useful functions
'''
def merge_bbox(bbox_list):
    min_left = float('inf')
    min_bottom = float('inf')
    max_right = float('-inf')
    max_top = float('-inf')

    for bbox in bbox_list:
        min_left = min(min_left, bbox.left)
        min_bottom = min(min_bottom, bbox.bottom)
        max_right = max(max_right, bbox.right)
        max_top = max(max_top, bbox.top)

    return BoundingBox(min_left, min_bottom, max_right, max_top)

def extract_sub_bbox_offsets(bbox, merged_bbox, RasterYSize, RasterXSize, redundant=1):
    """
    Calculate matrix indices based on bounding boxes.

    :param bbox: Original bounding box, in the form of BoundingBox(left=left, bottom=bottom, right=right, top=top)
    :param merged_bbox: Merged bounding box, in the form of BoundingBox(left=left, bottom=bottom, right=right, top=top)
    :param RasterYSize: Total number of rows in the raster
    :param RasterXSize: Total number of columns in the raster
    :param redundant: Additional pixels to include around the bounding box (default is 30)
    :return: Matrix indices (start_row, start_col, end_row, end_col)
    """
    # Calculate the positions of the top left and bottom right corners of the bounding box in the matrix
    start_col = min(max(int((bbox.left - merged_bbox.left) / (merged_bbox.right - merged_bbox.left) * RasterXSize) - redundant, 0), RasterXSize - 1)
    start_row = min(max(int((merged_bbox.top - bbox.top) / (merged_bbox.top - merged_bbox.bottom) * RasterYSize) - redundant, 0), RasterYSize - 1)
    end_col = max(min(int((bbox.right - merged_bbox.left) / (merged_bbox.right - merged_bbox.left) * RasterXSize) + redundant, RasterXSize - 1), 0)
    end_row = max(min(int((merged_bbox.top - bbox.bottom) / (merged_bbox.top - merged_bbox.bottom) * RasterYSize) + redundant, RasterYSize - 1), 0)

    return start_row, start_col, end_row, end_col

def extract_sub_data_from_sub_bbox(data,bbox_list,merged_bbox,redundant=1,nanmean_flag=True):
    # data is np.array([RasterXSize,RasterYSize])
    RasterYSize, RasterXSize = np.shape(data)
    sub_data_list = []
    for bbox in bbox_list:
        start_row, start_col, end_row, end_col = extract_sub_bbox_offsets(bbox, merged_bbox, RasterYSize, RasterXSize, redundant=redundant)
        if nanmean_flag:
            sub_data_list.append(np.nanmean(data[start_row:end_row,start_col:end_col]))
        else:
            sub_data_list.append(data[start_row:end_row,start_col:end_col])
    return sub_data_list

# 1. from polygon to bbox
def polygon2bbox(polygon):
    '''
    Function: Convert a polygon to a bounding box.
    Parameters:
        polygon (Polygon): The polygon to convert.
    Returns:
        BoundingBox: The bounding box of the polygon.
    '''
    # Get the bounding box of the polygon
    minx, miny, maxx, maxy = polygon.bounds
    # Create a BoundingBox for rasterio
    bbox = BoundingBox(left=minx, bottom=miny, right=maxx, top=maxy)
    return bbox

# 2. from bbox to polygon
def bbox2polygon(bbox):
    '''
    Function: Convert a bounding box to a polygon.
    Parameters:
        bbox (BoundingBox): The bounding box to convert.
    Returns:
        Polygon: The polygon representation of the bounding box.
    '''
    # Extract the coordinates from the bounding box
    minx, miny = bbox.left, bbox.bottom
    maxx, maxy = bbox.right, bbox.top
    
    # Create a polygon from the bounding box coordinates
    polygon = box(minx, miny, maxx, maxy)
    return polygon

# 3. Project bbox
def bbox2bbox(src_bbox,src_crs,dst_crs):
    """
    Convert a bounding box from one coordinate reference system (CRS) to another.

    Parameters:
        src_bbox (BoundingBox): The source bounding box to be transformed.
        src_crs (str or int): The source CRS of the bounding box, can be an EPSG code or a proj-string.
        dst_crs (str or int): The destination CRS to transform the bounding box to, can be an EPSG code or a proj-string.

    Returns:
        BoundingBox: The transformed bounding box in the destination CRS.
    """
    # Transform the coordinates of the bounding box
    # [left,right],[top,bottom] = transform(src_crs,dst_crs,[src_bbox.left,src_bbox.right],[src_bbox.top,src_bbox.bottom])
    left,bottom,right,top = transform_bounds(src_crs,dst_crs,*src_bbox)
    # Create and return the new bounding box
    bbox = BoundingBox(left=left, bottom=bottom, right=right, top=top)

    
    # bbox = transform_bounds(src_crs,dst_crs,*src_bbox)
    return bbox

def lonlat2polygon(max_lon,min_lon,min_lat,max_lat):
    polygon = Polygon([(max_lon, min_lat), (min_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat), (max_lon, min_lat)])
    return polygon

def bbox_to_pixel_offsets(geotransform, bbox, RasterYSize, RasterXSize, redundant=30):
    """
    Calculate matrix indices based on geotransform and bounding box.

    :param geotransform: Geotransform matrix, in the form of (top left X, pixel width, rotation, top left Y, rotation, pixel height)
    :param bbox: Bounding box, in the form of (minX, minY, maxX, maxY)
    :param RasterYSize: Total number of rows in the raster
    :param RasterXSize: Total number of columns in the raster
    :return: Matrix indices (start_row, start_col, end_row, end_col) and geotransform for the cropped area
    """
    # Extract elements from the geotransform matrix
    originX = geotransform[0]
    pixelWidth = geotransform[1]
    originY = geotransform[3]
    pixelHeight = geotransform[5]

    # Calculate the positions of the top left and bottom right corners of the bounding box in the matrix
    start_col = min(max(int((bbox[0] - originX) / pixelWidth) - redundant, 0),RasterXSize-1) # TODO using proper redundant 
    start_row = min(max(int((originY - bbox[3]) / -pixelHeight) - redundant, 0),RasterYSize-1)
    end_col = max(min(int((bbox[2] - originX) / pixelWidth) + redundant, RasterXSize-1), 0)
    end_row = max(min(int((originY - bbox[1]) / -pixelHeight) + redundant, RasterYSize-1), 0)

    return start_row, start_col, end_row, end_col

def get_sub_geotransform(geotransform,start_row,start_col,padding=0):
    # Extract elements from the geotransform matrix
    originX = geotransform[0]
    pixelWidth = geotransform[1]
    originY = geotransform[3]
    pixelHeight = geotransform[5]

    # Calculate the geotransform for the cropped area
    new_originX = originX + (start_col-padding) * pixelWidth
    new_originY = originY + (start_row-padding) * pixelHeight
    new_geotransform = (new_originX, pixelWidth, geotransform[2], new_originY, geotransform[4], pixelHeight)

    return new_geotransform

def geotransform_to_affine(geotransform):
    """
    Convert GDAL's geotransform to rasterio's affine format.

    :param geotransform: GDAL's geotransform, in the form of (top left X, pixel width, rotation, top left Y, rotation, pixel height)
    :return: rasterio's affine transformation
    """
    # 提取 geotransform 的元素
    originX = geotransform[0]
    pixelWidth = geotransform[1]
    rotationX = geotransform[2]
    originY = geotransform[3]
    rotationY = geotransform[4]
    pixelHeight = geotransform[5]

    # 创建 affine 变换
    affine = Affine(pixelWidth, rotationX, originX, rotationY, pixelHeight, originY)

    return affine

def affine_to_geotransform(affine):
    """
    Convert rasterio's affine format to GDAL's geotransform.

    :param affine: rasterio's affine transformation
    :return: GDAL's geotransform, in the form of (top left X, pixel width, rotation, top left Y, rotation, pixel height)
    """
    # 提取 affine 变换的元素
    pixelWidth = affine.a
    rotationX = affine.b
    originX = affine.c
    rotationY = affine.d
    pixelHeight = affine.e
    originY = affine.f

    # 创建 geotransform
    geotransform = (originX, pixelWidth, rotationX, originY, rotationY, pixelHeight)

    return geotransform

def geotransform_to_bbox(geotransform, width, height):
    """
    Convert GDAL geotransform to rasterio BoundingBox.

    Parameters:
    geotransform (tuple): GDAL geotransform tuple (x_origin, pixel_width, x_rotation, y_origin, y_rotation, pixel_height).
    width (int): Width of the raster in pixels.
    height (int): Height of the raster in pixels.

    Returns:
    BoundingBox: rasterio BoundingBox object.
    """
    x_origin, pixel_width, x_rotation, y_origin, y_rotation, pixel_height = geotransform

    # Calculate the coordinates of the bottom-left and top-right corners
    x_min = x_origin
    y_max = y_origin
    x_max = x_origin + (width * pixel_width) + (height * x_rotation)
    y_min = y_origin + (height * pixel_height) + (width * y_rotation)

    return BoundingBox(left=x_min, bottom=y_min, right=x_max, top=y_max)

def affine_to_bbox(affine, width, height):
    """
    Convert an affine transformation to a rasterio BoundingBox.

    Parameters:
    affine (Affine): Affine transformation object.
    width (int): Width of the raster in pixels.
    height (int): Height of the raster in pixels.

    Returns:
    BoundingBox: rasterio BoundingBox object.
    """
    # Calculate the coordinates of the bottom-left and top-right corners
    x_min, y_max = affine * (0, 0)  # Top-left corner
    x_max, y_min = affine * (width, height)  # Bottom-right corner

    return BoundingBox(left=x_min, bottom=y_min, right=x_max, top=y_max)


def expand_bbox(bbox,offset):
    # 原始边界框
    # 扩大边界框
    new_left = bbox.left - offset
    new_bottom = bbox.bottom - offset
    new_right = bbox.right + offset
    new_top = bbox.top + offset

    new_bbox = BoundingBox(new_left, new_bottom, new_right, new_top)

    return new_bbox

def get_width_height_from_bbox_affine(bbox,affine):
    # 从 Affine 变换中提取像素宽度和高度
    pixel_width = np.abs(affine.a)
    pixel_height = np.abs(affine.e)

    # 从 BoundingBox 中提取边界坐标
    x_min = bbox.left
    y_min = bbox.bottom
    x_max = bbox.right
    y_max = bbox.top

    # 计算宽度和高度
    width = int((x_max - x_min) / pixel_width)
    height = int((y_max - y_min) / pixel_height)
    return width,height

def bbox_to_geotransform(bbox, resolution):
    """
    Convert a rasterio BoundingBox to a GDAL geotransform tuple.

    Parameters:
    bbox (BoundingBox): rasterio BoundingBox object.
    resolution (float): Resolution of the raster in geographic units per pixel.

    Returns:
    tuple: GDAL geotransform tuple (x_origin, pixel_width, x_rotation, y_origin, y_rotation, pixel_height).
    """
    x_min, y_min, x_max, y_max = bbox.left, bbox.bottom, bbox.right, bbox.top

    # Calculate width and height in pixels
    width = (x_max - x_min) / resolution
    height = (y_max - y_min) / resolution

    # Create the geotransform tuple
    geotransform = (x_min, resolution, 0, y_max, 0, -resolution)

    return geotransform

def bbox_to_affine(bbox, resolution):
    """
    Convert a rasterio BoundingBox to an affine transformation.

    Parameters:
    bbox (BoundingBox): rasterio BoundingBox object.
    resolution (float): Resolution of the raster in geographic units per pixel.

    Returns:
    Affine: Affine transformation object.
    """
    x_min, y_min, x_max, y_max = bbox.left, bbox.bottom, bbox.right, bbox.top

    # Create the affine transformation
    affine = Affine.translation(x_min, y_max) * Affine.scale(resolution, -resolution)

    return affine