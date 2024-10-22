import re
from datetime import datetime
from pyproj import CRS
import numpy as np

aster_bands = ['VNIR_Swath:ImageData1','VNIR_Swath:ImageData2','VNIR_Swath:ImageData3',
         'SWIR_Swath:ImageData4','SWIR_Swath:ImageData5','SWIR_Swath:ImageData6',
         'SWIR_Swath:ImageData7','SWIR_Swath:ImageData8','SWIR_Swath:ImageData9']

ucc = np.matrix(([[0.676, 1.688, 2.25, 0.0], \
                [0.708, 1.415, 1.89, 0.0], \
                [0.423, 0.862, 1.15, 0.0], \
                [0.1087, 0.2174, 0.2900, 0.2900], \
                [0.0348, 0.0696, 0.0925, 0.4090], \
                [0.0313, 0.0625, 0.0830, 0.3900], \
                [0.0299, 0.0597, 0.0795, 0.3320], \
                [0.0209, 0.0417, 0.0556, 0.2450], \
                [0.0159, 0.0318, 0.0424, 0.2650]]))

# Thome et al. is used, which uses spectral irradiance values from MODTRAN
# Ordered b1, b2, b3N, b4, b5...b9
irradiance = [1848, 1549, 1114, 225.4, 86.63, 81.85, 74.85, 66.49, 59.85]

def parse_meta(meta):
    results = {
        'esd': None,
        'sza': None,
        'gain_dict': None,
        'utm_zone': None,
        'upperleft_y': None,
        'upperleft_x': None,
        'leftright_y': None,
        'leftright_x': None,
        'datetime': None,
        'solar_a': None,
        'solar_z': None,
        'satellite_z':None,
        'offset_dict':None,
        'sLatitude':None, 
        'sLongitude':None,
        'imagedatainfomation':None,
        'irradiance':irradiance
    }
    # Solar direction
    solar_info = meta["SOLARDIRECTION"].split(', ')
    # results['solar_a'] = 90-solar_info[0]
    results['solar_a'] = solar_info[0]
    results['solar_z'] = 90 - np.abs(np.float32(solar_info[1]))
    # Datetime
    Dateparm = meta["SETTINGTIMEOFPOINTING.1"].split('-')
    results['datetime'] = [int(Dateparm[0]), int(Dateparm[1]), int(Dateparm[2][:2])]

    # Calculate Earth-Sun Distance
    date = meta['CALENDARDATE']
    dated = datetime.strptime(date, '%Y%m%d')
    day = dated.timetuple()
    doy = day.tm_yday
    results['esd'] = 1.0 - 0.01672 * np.cos(np.radians(0.9856 * (doy - 4)))

    # Need SZA--calculate by grabbing solar elevation info
    results['sza'] = [np.float64(x) for x in meta['SOLARDIRECTION'].split(', ')][1]

    # Query gain data for each  band, needed for UCC
    gain_list = [g for g in meta.keys() if 'GAIN' in g]  ###### AARON HERE
    gain_info = []
    for f in range(len(gain_list)):
        gain_info1 = meta[gain_list[f]].split(', ')  # [0] ###### AARON HERE
        gain_info.append(gain_info1)
    results['gain_dict'] = dict(gain_info)

    satellite_info = [float(meta[f'POINTINGANGLE.{x}']) for x in range(1,4)]
    results['satellite_z'] = {"VNIR":satellite_info[0], "SWIR":satellite_info[1], "TIR":satellite_info[2]}

    offset_dict = {}
    for g in results['gain_dict'].keys():
        if g.split('.')[-1] in ['3N', '3B']:
            band = g.split('.')[-1]
        else:
            band = str(int(g.split('.')[-1]))
        offset_dict[g.split('.')[-1]] = meta[f'OFFSET{band}']
    results['offset_dict'] = offset_dict

    # Define UTM zone
    utm = np.int16(meta['UTMZONENUMBER'])
    n_s = np.float64(meta['NORTHBOUNDINGCOORDINATE'])

    # Create UTM zone code numbers
    utm_n = [i + 32600 for i in range(60)]
    utm_s = [i + 32700 for i in range(60)]

    # Define UTM zone based on North or South
    if n_s < 0:
        utm_zone = utm_s[utm]
    else:
        utm_zone = utm_n[utm]
    results['utm_zone'] = utm_zone

    # Define ul, lr
    ul = [np.float64(x) for x in meta['UPPERLEFTM'].split(', ')]
    lr = [np.float64(x) for x in meta['LOWERRIGHTM'].split(', ')]

    if n_s < 0:
        results['upperleft_y'] = ul[0] + 10000000
        results['upperleft_x'] = ul[1]

        results['leftright_y'] = lr[0] + 10000000
        results['leftright_x'] = lr[1]
    # Define extent for UTM North zones
    else:
        results['upperleft_y'] = ul[0]
        results['upperleft_x'] = ul[1]

        results['leftright_y'] = lr[0]
        results['leftright_x'] = lr[1]

    # Define center latitude and longitude
    sLatitude, sLongitude = GetCenter(meta)
    results['sLatitude'] = sLatitude
    results['sLongitude'] = sLongitude

    # Define image data information
    imagedatainfomation = {key: value for key, value in meta.items() if 'IMAGEDATAINFORMATION' in key}
    results['imagedatainfomation'] = imagedatainfomation
    return results

def get_transform(meta_parser,band_desc):
    match = re.search(r'ImageData([0-9]|10|11|12|13|14)', band_desc)
    if match:
        bn = int(match.group(1))
    if bn==3:
        bn_key = 'IMAGEDATAINFORMATION3N'
    else:
        bn_key = f'IMAGEDATAINFORMATION{str(bn)}'
    ncol = int(meta_parser['imagedatainfomation'][bn_key].split(', ')[1])
    nrow = int(meta_parser['imagedatainfomation'][bn_key].split(', ')[0])
    y_res = -1 * round((abs(meta_parser['upperleft_y'] - meta_parser['leftright_y'])) / ncol)
    x_res = round((abs(meta_parser['upperleft_x'] - meta_parser['leftright_x'])) / nrow)

    # Define UL x and y coordinates based on spatial resolution
    ul_yy = meta_parser['upperleft_y'] - (y_res / 2)
    ul_xx = meta_parser['upperleft_x'] - (x_res / 2)

    geotransform = (ul_xx, x_res, 0., ul_yy, 0., y_res)

    return geotransform
    
def get_width_height(meta_parser,band_desc):
    match = re.search(r'ImageData([0-9]|10|11|12|13|14)', band_desc)
    if match:
        bn = int(match.group(1))
    if bn==3:
        bn_key = 'IMAGEDATAINFORMATION3N'
    else:
        bn_key = f'IMAGEDATAINFORMATION{str(bn)}'
    ncol = int(meta_parser['imagedatainfomation'][bn_key].split(', ')[1])
    nrow = int(meta_parser['imagedatainfomation'][bn_key].split(', ')[0])
    return nrow,ncol

def get_projection(meta_parser):
    # 使用 pyproj 创建 CRS 对象
    crs = CRS.from_epsg(meta_parser['utm_zone'])
    # 导出为 WKT 格式
    projection = crs.to_wkt()
    return projection

def get_ucc1(meta_parser,band_desc):
    match = re.search(r'ImageData([0-9]|10|11|12|13|14)', band_desc)
    if match:
        bn = int(match.group(1))
    # ucc = meta_parser['ucc']
    gain_dict = meta_parser['gain_dict']

    if bn==3:
        bn_key = '3N'
    else:
        bn_key = f'0{str(bn)}'

    # Index start from 0 in python
    if gain_dict[bn_key] == 'HGH':
        ucc1 = ucc[bn-1, 0]
    elif gain_dict[bn_key] == 'NOR':
        ucc1 = ucc[bn-1, 1]
    elif gain_dict[bn_key] == 'LO1':
        ucc1 = ucc[bn-1, 2]
    else:
        ucc1 = ucc[bn-1, 3]
    return ucc1

def get_irradiance1(meta_parser,band_desc):
    match = re.search(r'ImageData([0-9]|10|11|12|13|14)', band_desc)
    if match:
        bn = int(match.group(1))
    irradiance = meta_parser['irradiance']
    # Index start from 0 in python
    irradiance1 = irradiance[bn-1]
    return irradiance1

def get_offset(meta_parser, band_desc):
    match = re.search(r'ImageData([0-9]|10|11|12|13|14)', band_desc)
    if match:
        bn = int(match.group(1))
    # ucc = meta_parser['ucc']
    offset_dict = meta_parser['offset_dict']

    if bn==3:
        bn_key = '3N'
    else:
        bn_key = f'0{str(bn)}'
    return float(offset_dict[bn_key])

def get_thetaz(meta_parser, band_desc):
    match = re.search(r'ImageData([0-9]|10|11|12|13|14)', band_desc)
    if match:
        bn = int(match.group(1))
    if bn<=3:
        bn_key = 'VNIR'
    elif bn<=9:
        bn_key='SWIR'
    else:
        bn_key='TIR'
    return float(meta_parser['satellite_z'][bn_key])

def get_k1k2(band_desc):
    k_vals = {
        'B10':{
            'K1': 3040.136402,
            'K2': 1735.337945
        },
        'B11':{
            'K1': 2482.375199,
            'K2': 1666.398761
        },
        'B12':{
            'K1': 1935.060183,
            'K2': 1585.420044
        },
        'B13':{
            'K1': 866.468575,
            'K2': 1350.069147
        },
        'B14':{
            'K1': 641.326517,
            'K2': 1271.221673
        }
    }
    match = re.search(r'ImageData(10|11|12|13|14)', band_desc)
    if match:
        bn = int(match.group(1))
    bn_key = f'B{str(bn)}'
    k1 = k_vals[bn_key]['K1']
    k2 = k_vals[bn_key]['K2']
    return k1,k2

def dn2rad(x,ucc1):
    #rad = np.where(x!=0, (x - 1.) * ucc1, 0)
    rad = (x - 1.) * ucc1
    return rad

def rad2ref(rad,esd,irradiance1,sea):
    ref = (np.pi * rad * (esd * esd)) / (irradiance1 * np.sin(np.pi * sea / 180))
    return ref

def dn2t(x,k1,k2):
    mask = x != 0  # 创建一个布尔数组，表示x中不为零的元素
    t = np.zeros_like(x)  # 创建一个与x形状相同的全零数组
    t[mask] = k2 / np.log(k1 / x[mask] + 1.) 
    return t

def GetCenter(metadata):
    point1lat = float(metadata["UPPERLEFT"].split(', ')[0])
    point1lon = float(metadata["UPPERLEFT"].split(', ')[1])
    point2lat = float(metadata["UPPERRIGHT"].split(', ')[0])
    point2lon = float(metadata["UPPERRIGHT"].split(', ')[1])
    point3lat = float(metadata["LOWERLEFT"].split(', ')[0])
    point3lon = float(metadata["LOWERLEFT"].split(', ')[1])
    point4lat = float(metadata["LOWERRIGHT"].split(', ')[0])
    point4lon = float(metadata["LOWERRIGHT"].split(', ')[1])
    sLongitude = (point1lon + point2lon + point3lon + point4lon) / 4
    sLatitude = (point1lat + point2lat + point3lat + point4lat) / 4
    return sLatitude, sLongitude
