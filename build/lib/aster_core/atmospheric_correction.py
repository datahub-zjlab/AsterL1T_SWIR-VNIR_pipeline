import numpy as np
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
import re

from aster_core.mosaic_tile import extract_granule,extract_geotif
from aster_core.utils import extract_sub_data_from_sub_bbox,merge_bbox
from aster_core.merge import merge_min
from aster_core.hdf_utils import parse_meta

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

solarz_list = [0, 2, 4, 6] + list(range(7, 42, 8)) + list(range(43, 60, 4)) + list(range(61, 70, 2)) + list(range(71, 81, 1))
aod_list = [2, 4, 6, 10, 20, 40, 70, 100, 150] + list(range(201, 801, 100)) + list(range(801, 2002, 400))
dem_list = [0, 200, 1000, 2000, 4000, 8000]

def py6s_coeffients(soz,soa,month,day,atmos_profile,band_id, dem_value, aod_value):
    from Py6S import SixS,Geometry,AeroProfile,GroundReflectance,Altitudes,Wavelength,PredefinedWavelengths,AtmosCorr
    s = SixS()
    s.geometry = Geometry.User()
    s.geometry.solar_z = float(soz)
    s.geometry.solar_a = float(soa)
    s.geometry.view_z = 0
    s.geometry.view_a = 0
    s.geometry.month = int(month)
    s.geometry.day = int(day)

    s.atmos_profile = atmos_profile
    # s.atmos_profile = AtmosProfile.UserWaterAndOzone(1,0.35) 

    s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.36)

    s.aot550 = aod_value * 0.001 

    s.altitudes = Altitudes()
    s.altitudes.set_target_custom_altitude(dem_value * 0.001)
    s.altitudes.set_sensor_satellite_level()

    band_mapping = {
        1: 'ASTER_B1',
        2: 'ASTER_B2',
        3: 'ASTER_B3N',
        4: 'ASTER_B4',
        5: 'ASTER_B5',
        6: 'ASTER_B6',
        7: 'ASTER_B7',
        8: 'ASTER_B8',
        9: 'ASTER_B9'
    }

    s.wavelength = Wavelength(getattr(PredefinedWavelengths, band_mapping[band_id]))

    s.atmos_corr = AtmosCorr.AtmosCorrBRDFFromRadiance(-0.1)
    # wavelengths, res = SixSHelpers.Wavelengths.run_wavelengths(s,[s.wavelength])
    s.run()
    a = s.outputs.coef_xa
    b = s.outputs.coef_xb
    c = s.outputs.coef_xc
    x = s.outputs.values

    return a, b, c

def find_nearest_neighbors(value_list, value):

    value_list.sort()  # 确保列表是有序的
    left = None
    right = None
    
    low = 0
    high = len(value_list) - 1
    
    while low <= high:
        mid = (low + high) // 2
        if value_list[mid] == value:
            left = value_list[mid - 1] if mid > 0 else None
            right = value_list[mid + 1] if mid < len(solarz_list) - 1 else None
            return left, right
        elif value_list[mid] < value:
            low = mid + 1
        else:
            high = mid - 1
    
    # 如果value不在列表中，找到最接近的两侧值
    if low == 0:
        left = value_list[0]
        right = value_list[1]
    elif low == len(value_list):
        left = value_list[-2]
        right = value_list[-1]
    else:
        left = value_list[low - 1]
        right = value_list[low]
    
    return left, right

def trilinear_interpolation(data,solar_z,aod,dem,solar_z_left,solar_z_right,aod_left,aod_right,dem_left,dem_right):
    points = []
    for d in data:
        points.append(d)
        
    V000 = next(p for p in points if p['solar_z'] == solar_z_left and p['aod'] == aod_left and p['dem'] == dem_left)
    V100 = next(p for p in points if p['solar_z'] == solar_z_right and p['aod'] == aod_left and p['dem'] == dem_left)
    V010 = next(p for p in points if p['solar_z'] == solar_z_left and p['aod'] == aod_right and p['dem'] == dem_left)
    V110 = next(p for p in points if p['solar_z'] == solar_z_right and p['aod'] == aod_right and p['dem'] == dem_left)
    V001 = next(p for p in points if p['solar_z'] == solar_z_left and p['aod'] == aod_left and p['dem'] == dem_right)
    V101 = next(p for p in points if p['solar_z'] == solar_z_right and p['aod'] == aod_left and p['dem'] == dem_right)
    V011 = next(p for p in points if p['solar_z'] == solar_z_left and p['aod'] == aod_right and p['dem'] == dem_right)
    V111 = next(p for p in points if p['solar_z'] == solar_z_right and p['aod'] == aod_right and p['dem'] == dem_right)

    x = (solar_z - solar_z_left) / (solar_z_right - solar_z_left)
    y = (aod - aod_left) / (aod_right - aod_left)
    z = (dem - dem_left) / (dem_right - dem_left)

    a = V000['a'] * (1 - x) * (1 - y) * (1 - z) + V100['a'] * x * (1 - y) * (1 - z) + V010['a'] * (1 - x) * y * (1 - z) + V110['a'] * x * y * (1 - z) + V001['a'] * (1 - x) * (1 - y) * z + V101['a'] * x * (1 - y) * z + V011['a'] * (1 - x) * y * z + V111['a'] * x * y * z
    b = V000['b'] * (1 - x) * (1 - y) * (1 - z) + V100['b'] * x * (1 - y) * (1 - z) + V010['b'] * (1 - x) * y * (1 - z) + V110['b'] * x * y * (1 - z) + V001['b'] * (1 - x) * (1 - y) * z + V101['b'] * x * (1 - y) * z + V011['b'] * (1 - x) * y * z + V111['b'] * x * y * z
    c = V000['c'] * (1 - x) * (1 - y) * (1 - z) + V100['c'] * x * (1 - y) * (1 - z) + V010['c'] * (1 - x) * y * (1 - z) + V110['c'] * x * y * (1 - z) + V001['c'] * (1 - x) * (1 - y) * z + V101['c'] * x * (1 - y) * z + V011['c'] * (1 - x) * y * z + V111['c'] * x * y * z

    return a,b,c

def retrieve_atmospheric_correction_paras(solar_z, solar_a, atmos_profile, aod, dem, band_id, month, day):
    with open(os.path.join(script_dir,'resource','Aster_6s_date_coeff.json'), 'r') as file:
        aster_6s_date_coeff = json.load(file)

    if aod < 5:
        aod = 5
     
    if dem < 5:
        dem = 5

    # retrieve from pgsql
    date = datetime(2020, month, day)
    start_date = datetime(2020, 1, 1)

    solar_z_left, solar_z_right = find_nearest_neighbors(solarz_list, solar_z)
    aod_left, aod_right = find_nearest_neighbors(aod_list, aod)
    dem_left, dem_right = find_nearest_neighbors(dem_list, dem)

    result_dict_upper = retrieve_6sparaslut(solar_z_left, solar_z_right, atmos_profile, aod_left, aod_right, dem_left, dem_right, band_id, date_flag='upper')
    result_dict_lower = retrieve_6sparaslut(solar_z_left, solar_z_right, atmos_profile, aod_left, aod_right, dem_left, dem_right, band_id, date_flag='lower')

    upper_a, upper_b, upper_c = trilinear_interpolation(result_dict_upper, solar_z, aod, dem, solar_z_left, solar_z_right, aod_left, aod_right, dem_left, dem_right)
    lower_a, lower_b, lower_c = trilinear_interpolation(result_dict_lower, solar_z, aod, dem, solar_z_left, solar_z_right, aod_left, aod_right, dem_left, dem_right)

    rough_a = lower_a + (upper_a - lower_a) * aster_6s_date_coeff[(date - start_date).days]
    rough_b = upper_b
    rough_c = upper_c

    return rough_a, rough_b, rough_c

def retrieve_6sparaslut(solar_z_left, solar_z_right, atmos_profile, aod_left, aod_right, dem_left, dem_right, band_id, date_flag='upper'):
    import sqlite3
    dbfile = os.path.join(script_dir,'resource','Aster_6s_lut_table.db')
    conn = sqlite3.connect(dbfile)

    if date_flag == 'upper':
        date = '2020-07-01'
    elif date_flag == 'lower':
        date = '2020-01-01'

    # 构建所有可能的组合
    conditions = []
    for solar_z in [solar_z_left, solar_z_right]:
        for aod in [aod_left, aod_right]:
            for dem in [dem_left, dem_right]:
                conditions.append((solar_z, aod, dem))

    # 构建SQL查询
    query = """
    SELECT soz, aod, dem, a, b, c
    FROM lut_table
    WHERE soz IN (?, ?) AND aod IN (?, ?) AND dem IN (?, ?)
    AND atmos_profile = ? AND band_id = ? AND date = ?
    """

    # 参数列表
    params = [solar_z_left, solar_z_right, aod_left, aod_right, dem_left, dem_right, atmos_profile, band_id, date]

    # 执行查询
    cursor = conn.execute(query, params)
    result = cursor.fetchall()

    # 构建返回的字典列表
    result_list = []
    for row in result:
        result_dict = {
            'solar_z': row[0],
            'aod': row[1],
            'dem': row[2],
            'a': row[3],
            'b': row[4],
            'c': row[5]
        }
        result_list.append(result_dict)

    return result_list

def get_aod_from_tile_bbox(hdf_list, tile_bbox, tile_crs, tile_size=64,nodata=-28672):
    tile_aod_list = []
    bands = ['Optical_Depth_055']
    for hdf_file in hdf_list:
        try:
            tile_aod,_ = extract_granule(hdf_file, bands, tile_bbox, tile_size, dst_crs=tile_crs)
            if not tile_aod is None:
                tile_aod = np.float32(tile_aod)
                tile_aod[tile_aod == nodata] = np.NaN
                if len(tile_aod.shape)==3:
                    tile_aod = np.nanmean(tile_aod, axis=0)
                tile_aod_list.append(tile_aod)
        except:
            continue

    if len(tile_aod_list)>1:
        merged_tile_aod = np.stack(tile_aod_list,axis=0)
        merged_tile_aod = np.nanmean(merged_tile_aod, axis=0)
        aod = np.nanmean(merged_tile_aod)
    elif len(tile_aod_list)==1:
        merged_tile_aod = tile_aod_list[0]
        aod = np.nanmean(tile_aod_list[0])
    else:
        aod = None
    return aod

def get_aod_from_tile_bbox_list(hdf_list, tile_bbox_list, tile_crs, tile_size=64,nodata=-28672):
    bands = ['Optical_Depth_055']
    merged_tile_bbox = merge_bbox(tile_bbox_list)
    tile_aod_list = []
    for hdf_file in hdf_list:
        try:
            tile_aod,meta = extract_granule(hdf_file, bands, tile_bbox=merged_tile_bbox, tile_size=tile_size, dst_crs=tile_crs)
            if not tile_aod is None:
                tile_aod = np.float32(tile_aod)
                tile_aod[tile_aod == nodata] = np.NaN
                tile_aod[tile_aod == 0] = np.NaN
                if len(np.shape(tile_aod))==3:
                    tile_aod = np.nanmean(tile_aod, axis=0)
                tile_aod_list.append(tile_aod)
        except:
            continue
    
    if len(tile_aod_list)>1:
        merged_tile_aod = np.stack(tile_aod_list,axis=0)
        merged_tile_aod = np.nanmean(merged_tile_aod, axis=0)
        aod_list = extract_sub_data_from_sub_bbox(merged_tile_aod,tile_bbox_list,merged_tile_bbox)
    elif len(tile_aod_list)==1:
        merged_tile_aod = tile_aod_list[0]
        aod_list = extract_sub_data_from_sub_bbox(merged_tile_aod,tile_bbox_list,merged_tile_bbox)
    else:
        aod_list = [np.nan]*len(tile_bbox_list)

    return aod_list

def get_default_aod(hdf_list,nodata=-28672):
    from osgeo import gdal
    bands = ['Optical_Depth_055']
    tile_aod_list = []
    for hdf_file in hdf_list:
        try:
            # tile_aod,meta = extract_granule(hdf_file, bands, tile_bbox=None, tile_size=tile_size)
            ds = gdal.Open(f'HDF4_EOS:EOS_GRID:{hdf_file}:grid1km:Optical_Depth_055')
            tile_aod = ds.ReadAsArray()
            if not tile_aod is None:
                tile_aod = np.float32(tile_aod)
                tile_aod[tile_aod == nodata] = np.NaN
                tile_aod[tile_aod == 0] = np.NaN
                if len(np.shape(tile_aod))==3:
                    tile_aod = np.nanmean(tile_aod, axis=0)
                tile_aod_list.append(tile_aod)
        except:
            continue

    if len(tile_aod_list)>1:
        merged_tile_aod = np.stack(tile_aod_list,axis=0)
        aod = np.nanmean(merged_tile_aod)
    elif len(tile_aod_list)==1:
        merged_tile_aod = tile_aod_list[0]
        aod = np.nanmean(merged_tile_aod)
    else:
        aod = None

    return aod

def get_dem_from_tile_bbox(geotiff_list, tile_bbox, tile_crs, tile_size=64):
    tile_dem_list = []
    for tiff_file in geotiff_list:
        try:
            tile_dem = extract_geotif(tiff_file, tile_bbox, tile_size, dst_crs=tile_crs)
            if not tile_dem is None:
                tile_dem_list.append(tile_dem)
        except:
            continue

    if len(tile_dem_list) > 1:
        merged_tile_dem = merge_min(tile_dem_list)
        dem = np.mean(merged_tile_dem)
    elif len(tile_dem_list)==1:
        merged_tile_dem = tile_dem_list[0]
        dem = np.mean(merged_tile_dem)
    else:
        merged_tile_dem = None
        dem = None
    return dem,merged_tile_dem

def get_dem_from_tile_bbox_list(geotiff_list, tile_bbox_list, tile_size=64, return_num_flag=False):
    tile_dem_list = []
    merged_tile_bbox = merge_bbox(tile_bbox_list)
    for tiff_file in geotiff_list:
        try:
            tile_dem = extract_geotif(tiff_file, merged_tile_bbox, tile_size, 'epsg:3857')
            if not tile_dem is None:
                tile_dem_list.append(tile_dem)
        except:
            continue

    if len(tile_dem_list) > 1:
        merged_tile_dem = merge_min(tile_dem_list)
        dem_tile_list = extract_sub_data_from_sub_bbox(merged_tile_dem,tile_bbox_list,merged_tile_bbox,nanmean_flag=False)

    elif len(tile_dem_list)==1:
        merged_tile_dem = tile_dem_list[0]
        dem_tile_list = extract_sub_data_from_sub_bbox(merged_tile_dem,tile_bbox_list,merged_tile_bbox,nanmean_flag=False)

    else:
        dem_list = [None]*len(tile_bbox_list)
    
    
    dem_list = [np.nanmean(dem_tile[dem_tile>0]) for dem_tile in dem_tile_list]
    dem_num_list = [np.sum(dem_tile>0) for dem_tile in dem_tile_list]

    if return_num_flag:
        return dem_list,dem_num_list

    return dem_list

def desc2id(band_desc):
    result = re.findall(r'\d+', band_desc)
    id = int(result[0])
    return id

def cal_atmospheric_correction_paras(meta,bands,aod,dem):
    from Py6S import AtmosProfile
    atmospheric_correction_paras = {}

    meta_parser = parse_meta(meta)
    solar_z = float(meta_parser['solar_z'])
    solar_a = float(meta_parser['solar_a'])
    year, month, day = meta_parser['datetime']
    sLatitude = meta_parser['sLatitude']
    atmos_profile = AtmosProfile.FromLatitudeAndDate(sLatitude, f"{int(year)}-{int(month)}-{int(day)}")

    for band in bands:
        atmospheric_correction_paras[band] = {}
        band_id = desc2id(band)
        a,b,c = py6s_coeffients(solar_z,solar_a,month,day,atmos_profile,band_id,dem,aod)
        atmospheric_correction_paras[band]['a'] = np.round(a,4)
        atmospheric_correction_paras[band]['b'] = np.round(b,4)
        atmospheric_correction_paras[band]['c'] = np.round(c,4)
    
    return atmospheric_correction_paras

def get_atmospheric_correction_paras(meta,bands,aod,dem):
    from Py6S import AtmosProfile
    atmospheric_correction_paras = {}
    meta_parser = parse_meta(meta)
    solar_z = float(meta_parser['solar_z'])
    solar_a = float(meta_parser['solar_a'])
    year, month, day = meta_parser['datetime']
    sLatitude = meta_parser['sLatitude']
    atmos_profile = AtmosProfile.FromLatitudeAndDate(sLatitude, f"{int(year)}-{int(month)}-{int(day)}")

    for band in bands:
        atmospheric_correction_paras[band] = {}
        band_id = desc2id(band)
        a,b,c = retrieve_atmospheric_correction_paras(solar_z,solar_a,atmos_profile,aod,dem,band_id,month,day)
        if (a is None) or (b is None) or (c is None):
            a,b,c = py6s_coeffients(solar_z,solar_a,month,day,atmos_profile,band_id,dem,aod)
        atmospheric_correction_paras[band]['a'] = np.round(a,5)
        atmospheric_correction_paras[band]['b'] = np.round(b,5)
        atmospheric_correction_paras[band]['c'] = np.round(c,5)
    return atmospheric_correction_paras

def atmospheric_correction_6s(radiance,bands,atmospheric_paras,nodata_value):
    reflectance_list = []
    for band_desc, sub_radiance in zip(bands, radiance):
        a,b,c = extract_atmospheric_correction_paras(atmospheric_paras, band_desc)
        reflectance = cal_atmospheric_correction_6s(sub_radiance,a,b,c,nodata_value=nodata_value)
        if reflectance is None:
                return None
        else:
            reflectance_list.append(reflectance)
    reflectance = np.stack(reflectance_list, axis=0)
    reflectance[np.isnan(reflectance)]=nodata_value
    reflectance[np.isinf(reflectance)]=nodata_value
    return reflectance

def extract_atmospheric_correction_paras(atmospheric_paras, band_desc):
    a = atmospheric_paras[band_desc]['a']
    b = atmospheric_paras[band_desc]['b']
    c = atmospheric_paras[band_desc]['c']
    return a,b,c

def cal_atmospheric_correction_6s(radiance, a, b, c, nodata_value=0):
    y = np.where(radiance != nodata_value, radiance * a - b, nodata_value)
    reflectance = np.where(y != nodata_value, y / (1 + y * c), nodata_value)
    return reflectance

