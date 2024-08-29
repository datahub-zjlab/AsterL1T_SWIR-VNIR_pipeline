import numpy as np

def common_used_functional_group(data):
    '''IRON'''
    ferric_iron = data[1] / data[0]
    ferrous_iron = data[4] / data[2] + data[0] / data[1]
    laterite = data[3] / data[4]
    gosson = data[3] / data[1]
    ferrous_silicates = data[4] / data[3]
    ferric_oxdes = data[3] / data[2]
    '''Carbonates'''
    carbonates = (data[6] + data[8]) / data[7]
    epidote = (data[5] + data[8]) / (data[6] + data[7])
    amphibole_mgoh = (data[5] + data[8]) / data[7]
    amphibole = data[5] / data[7]
    dolomit = (data[5] + data[7]) / data[6]
    '''Silicates'''
    sericite = (data[4] + data[6]) / data[5]
    alunite = (data[3] + data[5]) / data[4]
    phengitic = data[4] / data[5]
    muscovite = data[6] / data[5]
    clay = data[6] / data[4]
    alterarion = data[4] * data[6] / data[5] ** 2
    host_rock = data[3] / data[4]
    '''NDWI'''
    ndwi = ((data[0] - data[2]) / (data[0] + data[2]))
    ndvi = ((data[2] - data[1]) / (data[2] + data[1]))

    # 合并成一个矩阵
    result_matrix = np.squeeze(np.array([
        ferric_iron, ferrous_iron, laterite, gosson, ferrous_silicates, ferric_oxdes,
        carbonates, epidote, amphibole_mgoh, amphibole, dolomit, 
        sericite, alunite, phengitic, muscovite, clay, alterarion,host_rock, 
        ndwi, ndvi]
    ))

    return result_matrix
