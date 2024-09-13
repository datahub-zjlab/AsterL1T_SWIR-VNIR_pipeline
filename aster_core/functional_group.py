import numpy as np

functional_group_names = [
    'ferric_iron', 'ferrous_iron', 'laterite', 'gossan', 'ferrous_silicates', 'ferric_oxides',
    'carbonate_chlorite_epidote', 'mg_oh_alteration', 'amphibole_mgoh', 'amphibole', 'dolomite', 
    'sericite_muscovite_illite_smectite', 'alunite_kaolinite_pyrophyllite', 
    'phengitic', 'muscovite', 'kaolinite', 'clay', 'kaolinite_argillic', 'alunite_advanced_argillic', 
    'al_oh_alteration', 'calcite', 'ndwi', 'ndvi'
]

def common_used_functional_group(data):
    # 确保输入数据是三维的
    if data.ndim != 3 or data.shape[0] != 9:
        return None

    # 初始化结果矩阵
    # result_shape = (23, data.shape[1], data.shape[2])
    # result_matrix = np.zeros(result_shape)

    # 计算每个功能组的值
    ferric_iron = np.divide(data[1], data[0], out=np.full_like(data[0], np.nan), where=data[0] != 0)
    ferrous_iron = np.divide(data[4], data[2], out=np.full_like(data[2], np.nan), where=data[2] != 0) + np.divide(data[0], data[1], out=np.full_like(data[1], np.nan), where=data[1] != 0)
    laterite = np.divide(data[3], data[4], out=np.full_like(data[4], np.nan), where=data[4] != 0)
    gossan = np.divide(data[3], data[1], out=np.full_like(data[1], np.nan), where=data[1] != 0)
    ferrous_silicates = np.divide(data[4], data[3], out=np.full_like(data[3], np.nan), where=data[3] != 0)
    ferric_oxides = np.divide(data[3], data[2], out=np.full_like(data[2], np.nan), where=data[2] != 0)
    carbonate_chlorite_epidote = np.divide(data[6] + data[8], data[7], out=np.full_like(data[7], np.nan), where=data[7] != 0)
    mg_oh_alteration = np.divide(data[5] + data[8], data[6] + data[7], out=np.full_like(data[6] + data[7], np.nan), where=data[6] + data[7] != 0)
    amphibole_mgoh = np.divide(data[5] + data[8], data[7], out=np.full_like(data[7], np.nan), where=data[7] != 0)
    amphibole = np.divide(data[5], data[7], out=np.full_like(data[7], np.nan), where=data[7] != 0)
    dolomite = np.divide(data[5] + data[7], data[6], out=np.full_like(data[6], np.nan), where=data[6] != 0)
    # carbonate = np.divide(data[12], data[13], out=np.full_like(data[13], np.nan), where=data[13] != 0)
    # mafic_mineral = np.divide(data[11], data[12], out=np.full_like(data[12], np.nan), where=data[12] != 0) * np.divide(data[13], data[12], out=np.full_like(data[12], np.nan), where=data[12] != 0)
    sericite_muscovite_illite_smectite = np.divide(data[4] + data[6], data[5], out=np.full_like(data[5], np.nan), where=data[5] != 0)
    alunite_kaolinite_pyrophyllite = np.divide(data[3] + data[5], data[4], out=np.full_like(data[4], np.nan), where=data[4] != 0)
    phengitic = np.divide(data[4], data[5], out=np.full_like(data[5], np.nan), where=data[5] != 0)
    muscovite = np.divide(data[6], data[5], out=np.full_like(data[5], np.nan), where=data[5] != 0)
    kaolinite = np.divide(data[6], data[4], out=np.full_like(data[4], np.nan), where=data[4] != 0)
    clay = np.divide(data[4] * data[6], data[5] ** 2, out=np.full_like(data[5] ** 2, np.nan), where=data[5] != 0)
    kaolinite_argillic = np.divide(data[3], data[4], out=np.full_like(data[4], np.nan), where=data[4] != 0) * np.divide(data[7], data[5], out=np.full_like(data[5], np.nan), where=data[5] != 0)
    alunite_advanced_argillic = np.divide(data[6], data[4], out=np.full_like(data[4], np.nan), where=data[4] != 0) * np.divide(data[6], data[7], out=np.full_like(data[7], np.nan), where=data[7] != 0)
    al_oh_alteration = np.divide(data[3] * data[6], data[5] ** 2, out=np.full_like(data[5] ** 2, np.nan), where=data[5] != 0)
    # quartz_bearing_rock = np.divide(data[9], data[11], out=np.full_like(data[11], np.nan), where=data[11] != 0) * np.divide(data[12], data[11], out=np.full_like(data[11], np.nan), where=data[11] != 0)
    # quartz = np.divide(data[10] ** 2, data[9] * data[11], out=np.full_like(data[9] * data[11], np.nan), where=data[9] * data[11] != 0)
    calcite = np.divide(data[5] * data[8], data[7] ** 2, out=np.full_like(data[7] ** 2, np.nan), where=data[7] != 0)
    # garnet = np.divide(data[11] + data[13], data[12], out=np.full_like(data[12], np.nan), where=data[12] != 0)

    '''NDWI and NDVI'''
    ndwi = np.divide(data[0] - data[2], data[0] + data[2], out=np.full_like(data[0] + data[2], np.nan), where=data[0] + data[2] != 0)
    ndvi = np.divide(data[2] - data[1], data[2] + data[1], out=np.full_like(data[2] + data[1], np.nan), where=data[2] + data[1] != 0)

    # 合并成一个矩阵
    result_matrix = np.stack([
        ferric_iron, ferrous_iron, laterite, gossan, ferrous_silicates, ferric_oxides,
        carbonate_chlorite_epidote, mg_oh_alteration, amphibole_mgoh, amphibole, dolomite, 
        sericite_muscovite_illite_smectite, alunite_kaolinite_pyrophyllite, 
        phengitic, muscovite, kaolinite, clay, kaolinite_argillic, alunite_advanced_argillic, 
        al_oh_alteration, calcite, ndwi, ndvi
    ])

    # 去除NaN值和Inf值
    result_matrix[np.isnan(result_matrix) | np.isinf(result_matrix)] = 0

    return result_matrix