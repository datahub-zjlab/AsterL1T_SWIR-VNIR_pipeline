import numpy as np
import re

def restore_matrix_from_result(result, bands):
    """
    从result字典中恢复原始大小的矩阵
    
    :param result: 包含压缩数据的result字典
    :param bands: 波段列表
    :return: 恢复后的原始大小的矩阵
    """
    # 从result中获取边界信息
    min_row = result['min_row']
    min_col = result['min_col']
    max_row = result['max_row']
    max_col = result['max_col']
    match = re.search(r'res-(\d+)_tilesize-(\d+)', result['tile_info'])
    tile_size = int(match.group(2))
    
    # 创建一个与原始矩阵相同形状的全零矩阵
    restored_matrix = np.zeros((len(bands),tile_size,tile_size), dtype=np.uint8)
    
    # 从result中获取压缩后的数据
    compressed_data = []
    for band in bands:
        # hex_string = result[band].decode('utf-8')
        hex_string = result[band]
        byte_data = bytes.fromhex(hex_string)
        band_data = np.frombuffer(byte_data, dtype=np.uint8)
        compressed_data.append(band_data)
    
    # 将压缩后的数据转换为3D数组
    compressed_data = np.array(compressed_data)
    compressed_data = compressed_data.reshape((len(bands), max_row - min_row + 1, max_col - min_col + 1))
    
    # 将压缩后的数据填充回原始矩阵中
    restored_matrix[:, min_row:max_row+1, min_col:max_col+1] = compressed_data

    restored_matrix = np.squeeze(restored_matrix)
    
    return restored_matrix

def get_min_bounding_box(matrix, nodata_value=0):
    """
    获取包含所有非零值的最小边界框
    
    :param matrix: 输入矩阵
    :param nodata_value: 无效数据值，默认为0
    :return: 压缩后的矩阵和边界信息
    """
    if matrix.ndim == 2:
        rows, cols = matrix.shape
        min_row = next((i for i, row in enumerate(matrix) if any(row)), None)
        max_row = next((i for i, row in enumerate(matrix[::-1]) if any(row)), None)
        if max_row is not None:
            max_row = rows - 1 - max_row
        min_col = next((j for j, col in enumerate(matrix.T) if any(col)), None)
        max_col = next((j for j, col in enumerate(matrix.T[::-1]) if any(col)), None)
        if max_col is not None:
            max_col = cols - 1 - max_col
        if min_row is None or min_col is None or max_row is None or max_col is None:
            return matrix, (0, 0, rows, cols)
        return matrix[min_row:max_row+1, min_col:max_col+1], (min_row, min_col, max_row, max_col)
    elif matrix.ndim == 3:
        channels, rows, cols = matrix.shape
        all_mask = (matrix != nodata_value).astype(int)
        single_mask = np.prod(all_mask, axis=0)
        _, (min_row, min_col, max_row, max_col) = get_min_bounding_box(single_mask)
        return matrix[:, min_row:max_row+1, min_col:max_col+1], (min_row, min_col, max_row, max_col)

def matrix_to_byte(matrix):
    """
    将矩阵转换为二进制字节字符串
    
    :param matrix: 输入矩阵
    :return: 十六进制字符串表示的二进制数据
    """
    matrix_bytes = matrix.tobytes()
    hex_string = matrix_bytes.hex()
    if len(hex_string) % 2 != 0:
        hex_string = '0' + hex_string
    return hex_string


def transfer_matrix_to_odps_table_column(data,bands,zip_flag=False):
    result = {}
    if zip_flag:
        zip_data, bounding_box_info = get_min_bounding_box(data)
        zip_data = zip_data.astype(np.uint8)
        min_row, min_col, max_row, max_col = bounding_box_info

        result['min_row'] = min_row
        result['min_col'] = min_col
        result['max_row'] = max_row
        result['max_col'] = max_col

        for i, band in enumerate(bands):
            result[band] = matrix_to_byte(zip_data[i])
    else:
        for i, band in enumerate(bands):
            result[band] = matrix_to_byte(data[i])
    return result

def upload_data_to_odps(result_list, table, key_list=None):
    # Open a writer to write data to the table
    if key_list is None:
        key_list = result_list[0].keys()
    with table.open_writer(blocks=None) as writer:
        record_list = []
        for data in result_list:
            record = [data[key] for key in key_list]
            record_list.append(record)
        writer.write(record_list)