import os
import numpy as np
import re

# from rasterio.crs import CRS
from pyproj import CRS
from rasterio.transform import from_bounds, Affine
from rasterio.warp import reproject, Resampling

from aster_core.utils import bbox2bbox, bbox_to_pixel_offsets, geotransform_to_affine, get_sub_geotransform, geotransform_to_bbox,get_width_height_from_bbox_affine
from aster_core.hdf_utils import parse_meta,get_projection,get_transform

no_overlap_padding = 1

def read_data(data_path):
    from osgeo import gdal
    ds = gdal.Open(data_path)
    return ds

def extract_data_from_geotifs(tile_bbox, tile_size,tile_crs,
                             granule_file_list,
                             padding=0, redundant=100, 
                             return_dst_transform_flag=False):
    """
    Mosaic a tile from geotiff files based on the given bounding box and size.

    Args:
        tile_bbox (Boundingbox): Bounding box of the tile in EPSG:3857.
        tile_size (int): Size of the tile in pixels (tile_size x tile_size).
        granule_file_list (list): List of geotiff files to process.
        padding (int, optional): Padding around the tile. Default is 0.
        redundant (int, optional): Redundant pixels to include. Default is 100.

    Returns:
        List : Minimum reflectance mosaic of the tile.
        rasterio.Affine: dst_transform
    """

    ref_list = []
    dst_transform = from_bounds(*tile_bbox, tile_size, tile_size)
    
    for geotif_file in granule_file_list:
        merged_matrix = extract_geotif(geotif_file, tile_bbox=tile_bbox, tile_size=tile_size, dst_crs=tile_crs,
                                       padding=padding, redundant=redundant)

        if merged_matrix is not None:
            ref_list.append(merged_matrix)

    if return_dst_transform_flag:
        return ref_list, dst_transform
    else:
        return ref_list

def extract_data_from_hdfs(tile_bbox, tile_size, tile_crs, bands,
                          granule_file_list,
                          padding=0, redundant=100,
                          return_dst_transform_flag=False,
                          return_granule_id_flag=False):
    """
    Mosaic a tile from HDF files based on the given bounding box and size.

    Args:
        tile_bbox (Boundingbox): Bounding box of the tile in EPSG:3857.
        tile_size (int): Size of the tile in pixels (tile_size x tile_size).
        bands (list): List of bands to process.
        granule_file_list (list): List of HDF files to process.
        padding (int, optional): Padding around the tile. Default is 0.
        redundant (int, optional): Redundant pixels to include. Default is 100.

    Returns:
        np.ndarray: Minimum reflectance mosaic of the tile.
    """
    ref_list = []
    meta_list = []
    granule_id_list = []
    merge_data = None

    dst_transform = from_bounds(*tile_bbox, tile_size, tile_size)

    # start_time = time.time()
    for hdf_file in granule_file_list:
        # extract data from orig space to tile space (with predefined transform and projection)
        merged_matrix,meta = extract_granule(hdf_file, bands, tile_bbox, tile_size, dst_crs=tile_crs,
                                            padding=padding, redundant=redundant, 
                                            return_dst_transform_flag=False)
        

        if merged_matrix is not None:
            ref_list.append(merged_matrix)
            meta_list.append(meta)
            granule_id_list.append(os.path.basename(hdf_file).split('.')[0])

    if return_dst_transform_flag:
        if return_granule_id_flag:
            return ref_list, meta_list, granule_id_list, dst_transform
        else:
            return ref_list, meta_list, dst_transform
    else:
        if return_granule_id_flag:
            return ref_list, meta_list, granule_id_list
        else:
            return ref_list, meta_list
    

def extract_granule(hdf_file, bands, tile_bbox=None, tile_size=1024, dst_crs=None,
                    dst_transform=None, src_crs=None, ref_band=None,
                    padding=0, redundant=100, return_dst_transform_flag=False):
    """
    Extract and reproject data from an HDF file based on the given parameters.

    Args:
        hdf_file (str): Path to the HDF file.
        bands (list): List of bands to process.
        tile_bbox (Boundingbox): Bounding box of the tile in EPSG:3857.
        tile_size (int): Size of the tile in pixels (tile_size x tile_size).
        dst_transform (affine.Affine, optional): Destination transform for the tile.
        src_crs (rasterio.crs.CRS, optional): Destination projection.
        padding (int, optional): Padding around the tile. Default is 0.
        redundant (int, optional): Redundant pixels to include. Default is 100.
        return_dst_transform_flag (bool, optional): Whether to return the destination transform. Default is False.

    Returns:
        np.ndarray: Processed and reprojected matrix of the granule.
        dict: Granule meta.
        affine.Affine: Destination transform if return_dst_transform_flag is True.
    """
    try:
        hdf_ds = read_data(hdf_file)
        meta = hdf_ds.GetMetadata()
        sds = hdf_ds.GetSubDatasets()
        sds_data = {}

        if ref_band is None:
            ref_band = bands[-1]

        for sds_info in sds:
            sds_path = sds_info[0]
            match = re.search(ref_band, sds_path)
            if match:
                sub_ds = read_data(sds_path)
                width = sub_ds.RasterYSize
                height = sub_ds.RasterXSize
                
        if (tile_bbox is None) or (dst_crs is None):
            if sub_ds.GetProjection():
                projection = CRS.from_wkt(sub_ds.GetProjection())
                geotransform = sub_ds.GetGeoTransform()
            else:
                # TODO Only for ASTER L1T now
                meta_parser = parse_meta(meta)
                projection = get_projection(meta_parser)
                geotransform = get_transform(meta_parser, ref_band)

                # width,height = get_width_height(meta_parser,ref_band)
            
            dst_transform = geotransform
            tile_bbox = geotransform_to_bbox(dst_transform,width,height)
            dst_crs = projection

        else:
            if dst_transform is None:
                dst_transform = from_bounds(*tile_bbox, width=tile_size, height=tile_size)

        hdf_ds = None
        sub_ds = None

        # Process each subdataset
        for band_desc in bands:
            for sds_info in sds:
                sds_path = sds_info[0]
                match = re.search(band_desc, sds_path)
                if match:
                    dst_matrix = extract_subdataset(sds_path, tile_bbox, dst_crs, dst_transform, padding=padding, redundant=redundant)
                    if dst_matrix is not None:
                        sds_data[match.group(0)] = dst_matrix

        sorted_matrices = [sds_data[key] for key in sds_data.keys()]

        if len(sds_data) == len(bands):
            if len(sds_data) > 1:
                merged_matrix = np.stack(sorted_matrices, axis=0)
            elif len(sds_data) == 1:
                merged_matrix = sorted_matrices[0]
        else:
            # logger.error(f'Miss bands found in {hdf_file}')
            # print(f'Miss bands found in {hdf_file}')
            merged_matrix = None

        if return_dst_transform_flag:
            return merged_matrix, meta, dst_transform
        else:
            return merged_matrix, meta
    except:
        merged_matrix = None
        if return_dst_transform_flag:
            return merged_matrix, meta, dst_transform
        else:
            return merged_matrix, meta
        

def extract_geotif(geotif_file, tile_bbox=None, tile_size=1024, 
                   dst_crs=None, dst_transform=None, 
                   padding=0, redundant=100):
    """
    Extract and reproject data from a geotiff file based on the given parameters.

    Args:
        geotif_file (str): Path to the geotiff file.
        tile_bbox (Boundingbox): Bounding box of the tile in EPSG:3857, if None, using orig geotif bbox.
        tile_size (int): Size of the tile in pixels (tile_size x tile_size).
        dst_crs (str, optional): Destination CRS. Default is 'epsg:3857'.
        dst_transform (affine.Affine, optional): Destination transform for the tile.
        padding (int, optional): Padding around the tile. Default is 0.
        redundant (int, optional): Redundant pixels to include. Default is 100.

    Returns:
        np.ndarray: Processed and reprojected matrix of the geotiff.
    """
    ds = read_data(geotif_file)

    if ds is not None:

        projection = CRS.from_wkt(ds.GetProjection())
        geotransform = ds.GetGeoTransform()
        width = ds.RasterXSize
        height = ds.RasterYSize

        if (tile_bbox is None) or (dst_crs is None):

            dst_crs = projection
            dst_transform = geotransform
            tile_bbox = geotransform_to_bbox(geotransform,width,height)
            tile_bbox_dst = tile_bbox

            if not isinstance(dst_transform,Affine):
                dst_transform = geotransform_to_affine(dst_transform)
        
        else:

            if dst_transform is None:
                dst_transform = from_bounds(*tile_bbox, tile_size, tile_size)
            tile_bbox_dst = bbox2bbox(tile_bbox, dst_crs, projection)

        start_row, start_col, end_row, end_col = bbox_to_pixel_offsets(geotransform, tile_bbox_dst, ds.RasterYSize, ds.RasterXSize, redundant=redundant)
        sub_geotransform = get_sub_geotransform(geotransform, start_row, start_col, padding=padding)

        if (end_col - start_col) * (end_row - start_row) == 0:
            return None

        # Read the sub-array from the dataset
        sub_array = ds.ReadAsArray(start_col, start_row, end_col - start_col, end_row - start_row)
        if sub_array.ndim==3:
            sub_array = np.pad(sub_array, ((0,0),(no_overlap_padding,no_overlap_padding),(no_overlap_padding,no_overlap_padding)), mode='edge')
        else:
            sub_array = np.pad(sub_array, ((no_overlap_padding,no_overlap_padding),(no_overlap_padding,no_overlap_padding)), mode='edge')


        if np.count_nonzero(sub_array) == 0:
            return None

        # Reproject the reflectance data
        if sub_array.ndim == 3:
            dst_matrix = np.empty((sub_array.shape[0], tile_size + 2 * padding, tile_size + 2 * padding))
        else:
            dst_matrix = np.empty((tile_size + 2 * padding, tile_size + 2 * padding))

        dst_matrix, dst_transform = reproject(
            source=sub_array,
            destination=dst_matrix,
            src_transform=geotransform_to_affine(sub_geotransform),
            src_crs=projection,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )

        return dst_matrix

def extract_subdataset(sds_path, tile_bbox, dst_crs, dst_transform, padding=0, redundant=100):
    """
    Extract a subdataset from an HDF file to extract and reproject the data based on the given parameters.

    Args:
        sds_path (str): Path to the subdataset within the HDF file.
        tile_bbox (Boundingbox): Bounding box of the tile in EPSG:3857.
        tile_size (int): Size of the tile in pixels (tile_size x tile_size).
        dst_transform (affine.Affine): Destination transform for the tile.
        src_crs (rasterio.crs.CRS, optional): Destination projection.
        dst_crs (str, optional): Destination CRS. Default is 'epsg:3857'.
        padding (int, optional): Padding around the tile. Default is 0.
        redundant (int, optional): Redundant pixels to include. Default is 100.

    Returns:
        np.ndarray: Processed and reprojected matrix of the subdataset.
    """
    ds = read_data(sds_path)
    if ds is not None:
        # Get metadata from the dataset
        meta = ds.GetMetadata()
        band_desc = ds.GetDescription()

        if ds.GetProjection():
            projection = CRS.from_wkt(ds.GetProjection())
            geotransform = ds.GetGeoTransform()
        else:
            # TODO Only for ASTER L1T now
            meta_parser = parse_meta(meta)
            projection = get_projection(meta_parser)
            geotransform = get_transform(meta_parser, band_desc)

        tile_bbox_dst = bbox2bbox(tile_bbox, dst_crs, projection)

        if not isinstance(dst_transform,Affine):
            dst_transform = geotransform_to_affine(dst_transform)

        width,height = get_width_height_from_bbox_affine(tile_bbox,dst_transform)

        start_row, start_col, end_row, end_col = bbox_to_pixel_offsets(geotransform, tile_bbox_dst, ds.RasterYSize, ds.RasterXSize, redundant=redundant)
        sub_geotransform = get_sub_geotransform(geotransform, start_row, start_col, padding=padding)
        
        if (end_col - start_col) * (end_row - start_row) == 0:
            return None

        # Read the sub-array from the dataset
        sub_array = ds.ReadAsArray(start_col, start_row, end_col - start_col, end_row - start_row)

        ds = None

        if sub_array.ndim==3:
            sub_array = np.pad(sub_array, ((0,0),(no_overlap_padding,no_overlap_padding),(no_overlap_padding,no_overlap_padding)), mode='edge')
        else:
            sub_array = np.pad(sub_array, ((no_overlap_padding,no_overlap_padding),(no_overlap_padding,no_overlap_padding)), mode='edge')

        if np.count_nonzero(sub_array) == 0:
            return None

        # Reproject the reflectance data
        if sub_array.ndim == 3:
            dst_matrix = np.empty((sub_array.shape[0], height + 2 * padding, width + 2 * padding))
        else:
            dst_matrix = np.empty((height, width))

        # dst_matrix = np.squeeze(np.zeros((ds.RasterCount,tile_size, tile_size)))

        dst_matrix, dst_transform = reproject(
            source=sub_array,
            destination=dst_matrix,
            src_transform=geotransform_to_affine(sub_geotransform),
            src_crs=projection,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest  # Choose the resampling method
        )
        
        if np.count_nonzero(dst_matrix) == 0:
            return None
        
        return dst_matrix

