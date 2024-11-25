import os

import numpy as np
from datetime import datetime, timedelta
from rasterio.coords import BoundingBox
from aster_core.mosaic_tile import extract_data_from_hdfs,extract_data_from_geotifs,extract_granule,extract_geotif
from aster_core.preprocess import cal_radiance,cal_toa
from aster_core.atmospheric_correction import get_aod_from_tile_bbox,get_dem_from_tile_bbox,get_atmospheric_correction_paras,atmospheric_correction_6s
from aster_core.database import retrieve_files,retrieve_aod_files,retrieve_gdem_files
from aster_core.oss import download_file_from_oss
from aster_core.utils import bbox2bbox,bbox2polygon,geotransform_to_affine,affine_to_bbox
from aster_core.cloud import get_cloud_masks
from aster_core.merge import merge_deshadow_with_cloud_mask_nosort,merge_min,add_to_chanel
from aster_core.color_transfer import color_transfer
from aster_core.global_grid import modis_global_grid
from aster_core.functional_group import common_used_functional_group
from aster_core.odps import get_min_bounding_box,matrix_to_byte
from aster_core.hdf_utils import parse_meta,get_transform,get_projection,get_width_height

def tile_pipeline(tile_bbox,tile_size,tile_crs,bands,
                  aster_tmp_dir,aod_tmp_dir,dem_tmp_dir,modis_ref_tmp_dir,
                  aster_bucket_name='geocloud',
                  aod_bucket_name = 'aster-data-storage',
                  dem_bucket_name = 'aster-data-storage',
                  modis_ref_bucket_name = 'geocloud',
                  time_start='2000-01-01',time_end='2008-01-01',cloud_cover=30,aster_file_list=None):

    tile_region = bbox2polygon(bbox2bbox(tile_bbox,tile_crs,'epsg:4326'))

    if aster_file_list is None:
        result = retrieve_files(tile_region,time_start=time_start,time_end=time_end,cloud_cover=cloud_cover,download_flag=True)
        aster_file_list = []
        
        for granule_id in result.keys():
            try:
                hdf_file_url = result[granule_id]['file_url']
                hdf_file = os.path.join(aster_tmp_dir,os.path.basename(hdf_file_url))
                hdf_file = download_file_from_oss(hdf_file_url,bucket_name=aster_bucket_name,
                                                out_file=hdf_file,overwrite=False,oss_util_flag=False)
                aster_file_list.append(hdf_file)
            except:
                continue
        
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
        aster_toa = cal_toa(aster_radiance,meta,bands)
        if not aster_toa is None:
            aster_toa_list.append(aster_toa)
            tmp_meta_list.append(meta)
    meta_list = tmp_meta_list
    
    dem_result = retrieve_gdem_files(tile_region)
    dem_file_list = []
    for granule_id in dem_result.keys():
        try:
            tiff_file_url = f'fullgdem/{granule_id}_dem.tif'
            tiff_file = os.path.join(dem_tmp_dir,os.path.basename(tiff_file_url))
            tiff_file = download_file_from_oss(tiff_file_url,bucket_name=dem_bucket_name,
                                                out_file=tiff_file,overwrite=False,oss_util_flag=False)
            dem_file_list.append(tiff_file)
        except:
            continue
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
                try:
                    hdf_file_url = f'fullmodis/{granule_id}.hdf'
                    hdf_file = os.path.join(aod_tmp_dir,os.path.basename(hdf_file_url))
                    hdf_file = download_file_from_oss(hdf_file_url,bucket_name=aod_bucket_name,
                                                        out_file=hdf_file,overwrite=False,oss_util_flag=False)
                    aod_file_list.append(hdf_file)
                except:
                    continue
            aod = get_aod_from_tile_bbox(aod_file_list,tile_bbox,tile_crs,tile_size=64)
            if (not aod is None) or (aod is np.nan):
                atmospheric_correction_paras = get_atmospheric_correction_paras(meta,bands,aod,dem)
                reflectance = atmospheric_correction_6s(aster_radiance,bands,atmospheric_correction_paras,nodata_value=0)
                if (not reflectance is None) and (not np.count_nonzero(reflectance)==0):
                    aster_reflectance_list.append(reflectance)
    else:
        aster_reflectance_list = None
    
    if len(aster_reflectance_list)>1:
        reflectance_cloud_mask_list = get_cloud_masks(aster_reflectance_list)
        merge_ref_data,reflectance_merge_mask = merge_deshadow_with_cloud_mask_nosort(aster_reflectance_list,reflectance_cloud_mask_list,return_mask_in_chanel_flag=False)

    if len(aster_toa_list)>1:
        toa_cloud_mask_list = get_cloud_masks(aster_toa_list)
        merge_toa_data,toa_merge_mask = merge_deshadow_with_cloud_mask_nosort(aster_toa_list,toa_cloud_mask_list,return_mask_in_chanel_flag=False)

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
        modis_ref = None

    ct_data = color_transfer(merge_ref_data,modis_ref)

    result = common_used_functional_group(ct_data)
    
    if not result is None:
        result = add_to_chanel(result,reflectance_merge_mask)
    
    return merge_toa_data,ct_data,result

''' TODO ALREADY DONE IN CLASS TilePipeline
def download_aster_files(tile_region, time_start, time_end, cloud_cover, aster_tmp_dir, aster_bucket_name):
    result = retrieve_files(tile_region, time_start=time_start, time_end=time_end, cloud_cover=cloud_cover, download_flag=True)
    aster_file_list = []
    for granule_id in result.keys():
        try:
            hdf_file_url = result[granule_id]['file_url']
            hdf_file = os.path.join(aster_tmp_dir, os.path.basename(hdf_file_url))
            hdf_file = download_file_from_oss(hdf_file_url, bucket_name=aster_bucket_name,
                                              out_file=hdf_file, overwrite=False, oss_util_flag=False)
            aster_file_list.append(hdf_file)
        except:
            continue
    return aster_file_list

def process_aster_data(tile_bbox, tile_size, tile_crs, bands, aster_file_list):
    aster_dn_list, meta_list, granule_list = extract_data_from_hdfs(tile_bbox, tile_size, tile_crs, bands, aster_file_list, return_granule_id_flag=True)
    aster_radiance_list = []
    aster_toa_list = []
    tmp_meta_list = []
    tmp_granule_list = []

    for aster_dn, meta, granule_id in zip(aster_dn_list, meta_list, granule_list):
        aster_radiance = cal_radiance(aster_dn, meta, bands)
        if not aster_radiance is None:
            aster_radiance_list.append(aster_radiance)
            tmp_meta_list.append(meta)
            tmp_granule_list.append(granule_id)

    for aster_radiance, meta in zip(aster_radiance_list, tmp_meta_list):
        aster_toa = cal_toa(aster_radiance, meta, bands)
        if not aster_toa is None:
            aster_toa_list.append(aster_toa)

    return aster_radiance_list, aster_toa_list, tmp_meta_list, tmp_granule_list

def download_dem_files(tile_region, dem_tmp_dir, dem_bucket_name):
    dem_result = retrieve_gdem_files(tile_region)
    dem_file_list = []
    for granule_id in dem_result.keys():
        try:
            tiff_file_url = f'fullgdem/{granule_id}_dem.tif'
            tiff_file = os.path.join(dem_tmp_dir, os.path.basename(tiff_file_url))
            tiff_file = download_file_from_oss(tiff_file_url, bucket_name=dem_bucket_name,
                                               out_file=tiff_file, overwrite=False, oss_util_flag=False)
            dem_file_list.append(tiff_file)
        except:
            continue
    return dem_file_list

def process_dem_data(dem_file_list, tile_bbox, tile_crs, tile_size):
    dem = get_dem_from_tile_bbox(dem_file_list, tile_bbox, tile_crs, tile_size=64)
    return dem

def download_aod_files(tile_region, observation_time, aod_tmp_dir, aod_bucket_name):
    previous_day = observation_time - timedelta(days=0)
    next_day = observation_time + timedelta(days=1)
    aod_result = retrieve_aod_files(tile_region, time_start=previous_day, time_end=next_day)
    aod_file_list = []
    for granule_id in aod_result.keys():
        try:
            hdf_file_url = f'fullmodis/{granule_id}.hdf'
            hdf_file = os.path.join(aod_tmp_dir, os.path.basename(hdf_file_url))
            hdf_file = download_file_from_oss(hdf_file_url, bucket_name=aod_bucket_name,
                                              out_file=hdf_file, overwrite=False, oss_util_flag=False)
            aod_file_list.append(hdf_file)
        except:
            continue
    return aod_file_list

def process_aod_data(aod_file_list, tile_bbox, tile_crs, tile_size):
    aod = get_aod_from_tile_bbox(aod_file_list, tile_bbox, tile_crs, tile_size=64)
    return aod

def process_reflectance_data(aster_radiance_list, meta_list, granule_list, bands, aod, dem):
    aster_reflectance_list = []
    atmospheric_correction_paras_list = []
    for aster_radiance, meta, granule_id in zip(aster_radiance_list, meta_list, granule_list):
        atmospheric_correction_paras = get_atmospheric_correction_paras(meta, bands, aod, dem)
        reflectance = atmospheric_correction_6s(aster_radiance, bands, atmospheric_correction_paras, nodata_value=0)
        if (not reflectance is None) and (not np.count_nonzero(reflectance) == 0):
            aster_reflectance_list.append(reflectance)
            atmospheric_correction_paras_list.append(atmospheric_correction_paras)
    return aster_reflectance_list, atmospheric_correction_paras_list

def download_modis_ref_files(tile_bbox, modis_ref_tmp_dir, modis_ref_bucket_name):
    min_x, max_x, min_y, max_y = modis_global_grid.get_tile_index(tile_bbox)
    modis_ref_url_list = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            modis_ref_url_list.append(f'asterpreprocess/Modis_global_tiles/modis_res-500_tilesize-256_x-{x}_y-{y}_dst-deshadow.tiff')
    modis_ref_file_list = []
    for modis_ref_file_url in modis_ref_url_list:
        modis_ref_file = os.path.join(modis_ref_tmp_dir, os.path.basename(modis_ref_file_url))
        modis_ref_file = download_file_from_oss(modis_ref_file_url, bucket_name=modis_ref_bucket_name,
                                                out_file=modis_ref_file, overwrite=False, oss_util_flag=False)
        modis_ref_file_list.append(modis_ref_file)
    return modis_ref_file_list

def process_modis_ref_data(tile_bbox, tile_size, tile_crs, modis_ref_file_list):
    modis_ref_list = extract_data_from_geotifs(tile_bbox, tile_size, tile_crs, modis_ref_file_list)
    if len(modis_ref_list) > 1:
        modis_ref = merge_min(modis_ref_list)
    elif len(modis_ref_list) == 1:
        modis_ref = modis_ref_list[0]
    else:
        modis_ref = None
    return modis_ref
'''

class TilePipeline:
    def __init__(self, tile_bbox, tile_size, tile_crs, bands,
                 aster_tmp_dir, aod_tmp_dir, dem_tmp_dir, modis_ref_tmp_dir,
                 aster_bucket_name='geocloud',
                 aod_bucket_name='aster-data-storage',
                 dem_bucket_name='aster-data-storage',
                 modis_ref_bucket_name='geocloud',
                 time_start='2000-01-01', time_end='2008-01-01', cloud_cover=30, aster_file_list=None):
        self.tile_bbox = tile_bbox
        self.tile_size = tile_size
        self.tile_crs = tile_crs
        self.bands = bands
        self.aster_tmp_dir = aster_tmp_dir
        self.aod_tmp_dir = aod_tmp_dir
        self.dem_tmp_dir = dem_tmp_dir
        self.modis_ref_tmp_dir = modis_ref_tmp_dir
        self.aster_bucket_name = aster_bucket_name
        self.aod_bucket_name = aod_bucket_name
        self.dem_bucket_name = dem_bucket_name
        self.modis_ref_bucket_name = modis_ref_bucket_name
        self.time_start = time_start
        self.time_end = time_end
        self.cloud_cover = cloud_cover
        self.aster_file_list = aster_file_list

        self.tile_region = bbox2polygon(bbox2bbox(self.tile_bbox, self.tile_crs, 'epsg:4326'))
        self.aster_dn_list = []
        self.aster_toa_list = []
        self.atmospheric_correction_paras_list = []
        self.aster_reflectance_list = []
        self.reflectance_cloud_mask_list = []
        self.merge_toa_data = None
        self.merge_ref_data = None
        self.ct_data = None
        self.modis_ref = None
        self.result = None
        self.reflectance_meta_list = []
        self.reflectance_granule_list = []

    def download_aster_files(self):
        if self.aster_file_list is None:
            result = retrieve_files(self.tile_region, time_start=self.time_start, time_end=self.time_end, cloud_cover=self.cloud_cover, download_flag=True)
            self.aster_file_list = []
            for granule_id in result.keys():
                try:
                    hdf_file_url = result[granule_id]['file_url']
                    hdf_file = os.path.join(self.aster_tmp_dir, os.path.basename(hdf_file_url))
                    hdf_file = download_file_from_oss(hdf_file_url, bucket_name=self.aster_bucket_name,
                                                      out_file=hdf_file, overwrite=False, oss_util_flag=False)
                    self.aster_file_list.append(hdf_file)
                except:
                    continue

    def process_aster_data(self):
        self.aster_dn_list, meta_list, granule_list = extract_data_from_hdfs(self.tile_bbox, self.tile_size, self.tile_crs, self.bands, self.aster_file_list, return_granule_id_flag=True)
        aster_radiance_list = []
        aster_toa_list = []
        tmp_meta_list = []
        tmp_granule_list = []

        for aster_dn, meta, granule_id in zip(self.aster_dn_list, meta_list, granule_list):
            aster_radiance = cal_radiance(aster_dn, meta, self.bands)
            if not aster_radiance is None:
                aster_radiance_list.append(aster_radiance)
                tmp_meta_list.append(meta)
                tmp_granule_list.append(granule_id)

        for aster_radiance, meta in zip(aster_radiance_list, tmp_meta_list):
            aster_toa = cal_toa(aster_radiance, meta, self.bands)
            if not aster_toa is None:
                aster_toa_list.append(aster_toa)

        self.aster_radiance_list = aster_radiance_list
        self.aster_toa_list = aster_toa_list
        self.meta_list = tmp_meta_list
        self.granule_list = tmp_granule_list

    def download_dem_files(self):
        dem_result = retrieve_gdem_files(self.tile_region)
        dem_file_list = []
        for granule_id in dem_result.keys():
            try:
                tiff_file_url = f'fullgdem/{granule_id}_dem.tif'
                tiff_file = os.path.join(self.dem_tmp_dir, os.path.basename(tiff_file_url))
                tiff_file = download_file_from_oss(tiff_file_url, bucket_name=self.dem_bucket_name,
                                                   out_file=tiff_file, overwrite=False, oss_util_flag=False)
                dem_file_list.append(tiff_file)
            except:
                continue
        self.dem_file_list = dem_file_list

    def process_dem_data(self,tile_size=64):
        self.dem,self.tile_dem = get_dem_from_tile_bbox(self.dem_file_list, self.tile_bbox, self.tile_crs, tile_size=tile_size)

    def download_aod_files(self, observation_time):
        previous_day = observation_time - timedelta(days=0)
        next_day = observation_time + timedelta(days=1)
        aod_result = retrieve_aod_files(self.tile_region, time_start=previous_day, time_end=next_day)
        aod_file_list = []
        for granule_id in aod_result.keys():
            try:
                hdf_file_url = f'fullmodis/{granule_id}.hdf'
                hdf_file = os.path.join(self.aod_tmp_dir, os.path.basename(hdf_file_url))
                hdf_file = download_file_from_oss(hdf_file_url, bucket_name=self.aod_bucket_name,
                                                  out_file=hdf_file, overwrite=False, oss_util_flag=False)
                aod_file_list.append(hdf_file)
            except:
                continue
        return aod_file_list

    def process_aod_data(self, aod_file_list):
        self.aod = get_aod_from_tile_bbox(aod_file_list, self.tile_bbox, self.tile_crs, tile_size=64)

    def process_atmospheric_correction(self):
        for aster_radiance, meta, granule_id in zip(self.aster_radiance_list, self.meta_list, self.granule_list):
            observation_time = datetime.strptime(meta['SETTINGTIMEOFPOINTING.1'], '%Y-%m-%dT%H:%M:%SZ')
            aod_file_list = self.download_aod_files(observation_time)
            self.process_aod_data(aod_file_list)
            if (not self.aod is None) and (not self.aod is np.nan):
                atmospheric_correction_paras = get_atmospheric_correction_paras(meta, self.bands, self.aod, self.dem)
                reflectance = atmospheric_correction_6s(aster_radiance, self.bands, atmospheric_correction_paras, nodata_value=0)
                if (not reflectance is None) and (not np.count_nonzero(reflectance) == 0):
                    self.aster_reflectance_list.append(reflectance)
                    self.atmospheric_correction_paras_list.append(atmospheric_correction_paras)
                    self.reflectance_meta_list.append(meta)
                    self.reflectance_granule_list.append(granule_id)

    def process_merge(self):
        if len(self.aster_reflectance_list) > 1:
            self.reflectance_cloud_mask_list = get_cloud_masks(self.aster_reflectance_list)
            self.merge_ref_data, self.reflectance_merge_mask = merge_deshadow_with_cloud_mask_nosort(self.aster_reflectance_list, self.reflectance_cloud_mask_list, return_mask_in_chanel_flag=False)

        if len(self.aster_toa_list) > 1:
            toa_cloud_mask_list = get_cloud_masks(self.aster_toa_list)
            self.merge_toa_data, self.toa_merge_mask = merge_deshadow_with_cloud_mask_nosort(self.aster_toa_list, toa_cloud_mask_list, return_mask_in_chanel_flag=False)

    def download_modis_ref_files(self):
        min_x, max_x, min_y, max_y = modis_global_grid.get_tile_index(self.tile_bbox)
        modis_ref_url_list = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                modis_ref_url_list.append(f'asterpreprocess/Modis_global_tiles/modis_res-500_tilesize-256_x-{x}_y-{y}_dst-deshadow.tiff')
        modis_ref_file_list = []
        for modis_ref_file_url in modis_ref_url_list:
            modis_ref_file = os.path.join(self.modis_ref_tmp_dir, os.path.basename(modis_ref_file_url))
            modis_ref_file = download_file_from_oss(modis_ref_file_url, bucket_name=self.modis_ref_bucket_name,
                                                    out_file=modis_ref_file, overwrite=False, oss_util_flag=False)
            modis_ref_file_list.append(modis_ref_file)
        self.modis_ref_file_list = modis_ref_file_list

    def process_modis_ref_data(self):
        modis_ref_list = extract_data_from_geotifs(self.tile_bbox, self.tile_size, self.tile_crs, self.modis_ref_file_list)
        if len(modis_ref_list) > 1:
            self.modis_ref = merge_min(modis_ref_list)
        elif len(modis_ref_list) == 1:
            self.modis_ref = modis_ref_list[0]
        else:
            self.modis_ref = None

    def process_color_transfer(self):
        self.ct_data = color_transfer(self.merge_ref_data, self.modis_ref)

    def process_functional_group(self):
        self.result = common_used_functional_group(self.ct_data)
        if not self.result is None:
            self.result = add_to_chanel(self.result, self.reflectance_merge_mask)

    def run(self):
        self.download_aster_files()
        self.process_aster_data()
        self.download_dem_files()
        self.process_dem_data()
        if (not self.dem is None) and (not self.dem is np.nan):
            self.process_atmospheric_correction()
        self.process_merge()
        self.download_modis_ref_files()
        self.process_modis_ref_data()
        self.process_color_transfer()
        self.process_functional_group()

    def get_results(self):
        return {
            'aster_dn_list': self.aster_dn_list,
            'aster_toa_list': self.aster_toa_list,
            'atmospheric_correction_paras_list': self.atmospheric_correction_paras_list,
            'aster_reflectance_list': self.aster_reflectance_list,
            'reflectance_cloud_mask_list': self.reflectance_cloud_mask_list,
            'merge_toa_data': self.merge_toa_data,
            'merge_ref_data': self.merge_ref_data,
            'ct_data': self.ct_data,
            'modis_ref': self.modis_ref,
            'result': self.result,
            'reflectance_meta_list': self.reflectance_meta_list,
            'reflectance_granule_list': self.reflectance_granule_list
        }

def odps_pipeline(aster_dn, meta, bands, atmospheric_paras, nodata_value=0):
    aster_radiance = cal_radiance(aster_dn,meta,bands,nodata_value=nodata_value)
    aster_toa = cal_toa(aster_radiance,meta,bands,nodata_value=nodata_value)
    aster_reflectance = atmospheric_correction_6s(aster_radiance,bands,atmospheric_paras,nodata_value=nodata_value)
    return aster_radiance,aster_toa,aster_reflectance

def process_tile(tile_index, input_file, global_grid, bands=None):
    
    tile_bbox = global_grid.get_tile_bounds(tile_index)

    if 'tif' in input_file:
        data = extract_geotif(input_file, tile_bbox, global_grid.tile_size, global_grid.projection)

    elif 'hdf' in input_file:
        data,meta = extract_granule(input_file, bands, tile_bbox, global_grid.tile_size, global_grid.projection)

    result = {}
    tile_index_x, tile_index_y = tile_index
    result['tile_index_x'] = tile_index_x
    result['tile_index_y'] = tile_index_y

    if data is not None:
        zip_data, bounding_box_info = get_min_bounding_box(data)
        # zip_data[zip_data<0]=1
        zip_data = zip_data.astype(np.uint8)

        if len(zip_data.shape)==2:
            zip_data = np.expand_dims(zip_data,0)

        min_row, min_col, max_row, max_col = bounding_box_info

        result['min_row'] = min_row
        result['min_col'] = min_col
        result['max_row'] = max_row
        result['max_col'] = max_col
        result['tile_index_x'] = tile_index_x
        result['tile_index_y'] = tile_index_y
        result['tile_info'] = f'res-{global_grid.resolution}_tilesize-{global_grid.tile_size}'
        
        for i, band in enumerate(bands):
            result[band] = matrix_to_byte(zip_data[i])

    return result

def get_bbox_from_geotiff(geotiff_path):
    import rasterio
    """
    从GeoTIFF文件中获取边界框（Bounding Box）

    :param geotiff_path: GeoTIFF文件的路径
    :return: rasterio.coords.BoundingBox对象
    """
    with rasterio.open(geotiff_path) as src:
        # 获取图像的宽度和高度
        width = src.width
        height = src.height

        # 获取图像的变换矩阵
        transform = src.transform

        # 计算边界框的四个角点坐标
        left = transform.c
        top = transform.f
        right = left + transform.a * width
        bottom = top + transform.e * height

        # 创建BoundingBox对象
        bbox = BoundingBox(left=left, bottom=bottom, right=right, top=top)
        return bbox
    
def get_bbox_from_aster(hdf_file, dst_crs='epsg:3857'):
    from osgeo import gdal
    """
    从ASTER HDF文件中获取边界框信息，并转换为目标坐标系
    
    :param hdf_file: ASTER HDF文件路径
    :param dst_crs: 目标坐标系，默认为'epsg:3857'
    :return: 转换后的边界框
    """
    ds = gdal.Open(hdf_file)
    meta = ds.GetMetadata()
    meta_parser = parse_meta(meta)
    projection = get_projection(meta_parser)
    geotransform = get_transform(meta_parser, 'ImageData1')
    affine = geotransform_to_affine(geotransform)
    width,height = get_width_height(meta_parser,'ImageData1')
    bbox = affine_to_bbox(affine, width, height)
    dst_bbox = bbox2bbox(bbox, projection, dst_crs)
    return dst_bbox

def transfer_data_to_odps_list(input_file,bbox,global_grid,bands=None):
    tile_index_list = global_grid.get_tile_list(bbox)
    result_list = []
    for tile_index in tile_index_list:
        result = process_tile(tile_index,input_file,global_grid,bands=bands)
        if not result == {}:
            result_list.append(result)
    return result_list


    
