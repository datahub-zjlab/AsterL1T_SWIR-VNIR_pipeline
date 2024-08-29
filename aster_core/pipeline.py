import os
from datetime import datetime, timedelta
import numpy as np

from aster_core.mosaic_tile import extract_data_from_hdfs,extract_data_from_geotifs
from aster_core.preprocess import cal_radiance,cal_toa
from aster_core.atmospheric_correction import get_aod_from_tile_bbox,get_dem_from_tile_bbox,get_atmospheric_correction_paras,atmospheric_correction_6s
from aster_core.database import retrieve_files,retrieve_aod_files,retrieve_gdem_files
from aster_core.oss import download_file_from_oss
from aster_core.utils import bbox2bbox,bbox2polygon
from aster_core.cloud import get_cloud_masks
from aster_core.merge import merge_deshadow_with_cloud_mask,merge_min,add_to_chanel
from aster_core.color_transfer import color_transfer
from aster_core.global_grid import modis_global_grid
from aster_core.functional_group import common_used_functional_group

def tile_pipeline(tile_bbox,tile_size,tile_crs,bands,
                  aster_tmp_dir,aod_tmp_dir,dem_tmp_dir,modis_ref_tmp_dir,
                  aster_bucket_name='geocloud',
                  aod_bucket_name = 'aster-data-storage',
                  dem_bucket_name = 'aster-data-storage',
                  modis_ref_bucket_name = 'geocloud',
                  time_start='2000-01-01',time_end='2008-01-01',cloud_cover=30):

    tile_region = bbox2polygon(bbox2bbox(tile_bbox,tile_crs,'epsg:4326'))
    result = retrieve_files(tile_region,time_start=time_start,time_end=time_end,cloud_cover=cloud_cover,download_flag=True)
    aster_file_list = []
    for granule_id in result.keys():
        hdf_file_url = result[granule_id]['file_url']
        hdf_file = os.path.join(aster_tmp_dir,os.path.basename(hdf_file_url))
        hdf_file = download_file_from_oss(hdf_file_url,bucket_name=aster_bucket_name,
                                          out_file=hdf_file,overwrite=False,oss_util_flag=False)
        aster_file_list.append(hdf_file)
        
    aster_dn_list,meta_list = extract_data_from_hdfs(tile_bbox,tile_size,tile_crs,bands,aster_file_list)
    aster_radiance_list = []
    aster_toa_list = []
    aster_reflectance_list = []

    tmp_meta_list = []
    for aster_dn,meta in zip(aster_dn_list,meta_list):
        aster_radiance = cal_radiance(aster_dn,meta,bands)
        if not aster_radiance is None:
            aster_radiance_list.append(aster_radiance)
            tmp_meta_list.append(meta)
    meta_list = tmp_meta_list
            
    tmp_meta_list = []
    for aster_radiance,meta in zip(aster_radiance_list,meta_list):
        aster_toa = cal_toa(aster_dn,meta,bands)
        if not aster_toa is None:
            aster_toa_list.append(aster_toa)
            tmp_meta_list.append(meta)
    meta_list = tmp_meta_list
    
    dem_result = retrieve_gdem_files(tile_region)
    dem_file_list = []
    for granule_id in dem_result.keys():
        tiff_file_url = f'fullgdem/{granule_id}_dem.tif'
        tiff_file = os.path.join(dem_tmp_dir,os.path.basename(tiff_file_url))
        tiff_file = download_file_from_oss(tiff_file_url,bucket_name=dem_bucket_name,
                                            out_file=tiff_file,overwrite=False,oss_util_flag=False)
        dem_file_list.append(tiff_file)
    dem = get_dem_from_tile_bbox(dem_file_list,tile_bbox,tile_crs,tile_size=64)
    
    if (not dem is None) or (not dem is np.nan):
        for aster_radiance,meta in zip(aster_radiance_list,meta_list):
            date_format = '%Y-%m-%dT%H:%M:%SZ'
            observation_time = datetime.strptime(meta['SETTINGTIMEOFPOINTING.1'],date_format)
            previous_day = observation_time - timedelta(days=0)
            next_day = observation_time + timedelta(days=1)
            aod_result = retrieve_aod_files(tile_region,time_start=previous_day,time_end=next_day)

            aod_file_list = []
            for granule_id in aod_result.keys():
                hdf_file_url = f'fullmodis/{granule_id}.hdf'
                hdf_file = os.path.join(aod_tmp_dir,os.path.basename(hdf_file_url))
                hdf_file = download_file_from_oss(hdf_file_url,bucket_name=aod_bucket_name,
                                                    out_file=hdf_file,overwrite=False,oss_util_flag=False)
                aod_file_list.append(hdf_file)
            aod = get_aod_from_tile_bbox(aod_file_list,tile_bbox,tile_crs,tile_size=64)
            if (not aod is None) or (aod is np.nan):
                atmospheric_correction_paras = get_atmospheric_correction_paras(meta,bands,aod,dem)
                reflectance = atmospheric_correction_6s(aster_radiance,bands,atmospheric_correction_paras,nodata_value=0)
                if not reflectance is None:
                    aster_reflectance_list.append(reflectance)
    else:
        aster_reflectance_list = None
    
    if len(aster_reflectance_list)>1:
        cloud_mask_list = get_cloud_masks(aster_reflectance_list)
        merge_data,merge_mask = merge_deshadow_with_cloud_mask(aster_reflectance_list,cloud_mask_list,return_mask_in_chanel_flag=False)

    min_x, max_x, min_y, max_y = modis_global_grid.get_tile_index(tile_bbox)
    modis_ref_url_list = []
    for x in range(min_x,max_x+1):
        for y in range(min_y,max_y+1):
            modis_ref_url_list.append(f'asterpreprocess/Modis_global_tiles/modis_res-500_tilesize-256_x-{x}_y-{y}_dst-deshadow.tiff')

    modis_ref_file_list = []
    for modis_ref_file_url in modis_ref_url_list:
        
        modis_ref_file = os.path.join(modis_ref_tmp_dir,os.path.basename(modis_ref_file_url))
        modis_ref_file = download_file_from_oss(modis_ref_file_url,bucket_name=modis_ref_bucket_name,
                                            out_file=modis_ref_file,overwrite=False,oss_util_flag=False)
        modis_ref_file_list.append(modis_ref_file)
    modis_ref_list = extract_data_from_geotifs(tile_bbox,tile_size,tile_crs,modis_ref_file_list)
    if len(modis_ref_list) > 1:
        modis_ref = merge_min(modis_ref_list)
    elif len(modis_ref_list)==1:
        modis_ref = modis_ref_list[0]
    else:
        dem = None
    data = color_transfer(merge_data,modis_ref)

    result = common_used_functional_group(data)
    result = add_to_chanel(result,merge_mask)
    return result

def odps_pipeline(aster_dn, meta, bands, atmospheric_paras, nodata_value=0):
    aster_radiance = cal_radiance(aster_dn,meta,bands,nodata_value=nodata_value)
    aster_toa = cal_toa(aster_radiance,meta,bands,nodata_value=nodata_value)
    aster_reflectance = atmospheric_correction_6s(aster_radiance,bands,atmospheric_paras,nodata_value=nodata_value)



    


    
