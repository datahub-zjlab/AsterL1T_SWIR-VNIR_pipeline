import os
import numpy as np
from scipy.ndimage import zoom
from PIL import Image
from aster_core.tile_config import GlobalMercator, MAXZOOMLEVEL

class BaseTileGenerator:
    '''
    Only works in 'epsg:3857' or 'mercator' profile!
    '''
    def __init__(self, data, geotransform, bbox, max_zoom=None, min_zoom=3, nodata_value=0, tile_size=256, xyz_flag=True, max_zoom_level=MAXZOOMLEVEL):
        """
        Initialize the BaseTileGenerator.

        :param data: Input data array with shape (raster_count, x_size, y_size).
        :param geotransform: GDAL geotransform tuple.
        :param bbox: Bounding box of the data.
        :param max_zoom: Maximum zoom level. If None, it will be calculated based on the pixel size.
        :param min_zoom: Minimum zoom level.
        :param nodata_value: No data value.
        :param tile_size: Size of each tile.
        :param xyz_flag: Flag to indicate if XYZ tile format is used.
        :param max_zoom_level: Maximum allowed zoom level.
        """
        self.nodata_value = nodata_value
        self.tile_size = tile_size
        self.max_zoom_level = max_zoom_level
        self.xyz_flag = xyz_flag

        self.data = data
        self.raster_count, self.y_size, self.x_size = self.data.shape
        self.geotransform = geotransform
        self.bbox = bbox

        self.mercator = GlobalMercator(tile_size=self.tile_size)

        if max_zoom is None:
            self.max_zoom = self.mercator.ZoomForPixelSize(self.geotransform[1])
        else:
            self.max_zoom = min(max_zoom, self.mercator.ZoomForPixelSize(self.geotransform[1]))
        
        self.min_zoom = min(min_zoom, self.max_zoom)

        self.min_tile_x, self.min_tile_y, self.max_tile_x, self.max_tile_y = self.generate_tile_coordinates()

    def generate_tile_coordinates(self):
        """
        Generate tile coordinates for the given zoom level.

        :return: Tuple of (min_tile_x, min_tile_y, max_tile_x, max_tile_y).
        """
        min_x = self.geotransform[0]
        max_x = self.geotransform[0] + self.x_size * self.geotransform[1]
        max_y = self.geotransform[3]
        min_y = self.geotransform[3] - self.y_size * self.geotransform[1]
        
        zoom_level = self.max_zoom
        min_tile_x, min_tile_y = self.mercator.MetersToTile(min_x, min_y, zoom_level)
        max_tile_x, max_tile_y = self.mercator.MetersToTile(max_x, max_y, zoom_level)
        min_tile_x, min_tile_y = max(0, min_tile_x), max(0, min_tile_y)
        max_tile_x, max_tile_y = min(2**zoom_level - 1, max_tile_x), min(2**zoom_level - 1, max_tile_y)

        return (min_tile_x, min_tile_y, max_tile_x, max_tile_y)

    def get_y_tile(self, tms_y, zoom_level):
        """
        Convert TMS tile coordinates to XYZ tile coordinates.

        :param tms_y: TMS y coordinate.
        :param zoom_level: Zoom level.
        :return: XYZ y coordinate.
        """
        if self.xyz_flag:
            return (2**zoom_level - 1) - tms_y
        return tms_y

    def geo_query(self, bounds, query_size=None):
        """
        Perform a geographic query to get the raster data within the given bounds.

        :param bounds: Bounding box in geographic coordinates.
        :param query_size: Size of the query area.
        :return: Tuple of (raster_bounds, window_bounds).
        """
        geo_transform = self.geotransform
        raster_x = int((bounds[0] - geo_transform[0]) / geo_transform[1] + 0.001)
        raster_y = int((bounds[3] - geo_transform[3]) / geo_transform[5] + 0.001)
        raster_width = max(1, int((bounds[2] - bounds[0]) / geo_transform[1] + 0.5))
        raster_height = max(1, int((bounds[1] - bounds[3]) / geo_transform[5] + 0.5))

        if not query_size:
            window_width, window_height = raster_width, raster_height
        else:
            window_width, window_height = query_size, query_size

        window_x = 0
        if raster_x < 0:
            x_shift = abs(raster_x)
            window_x = int(window_width * (float(x_shift) / raster_width))
            window_width = window_width - window_x
            raster_width = raster_width - int(raster_width * (float(x_shift) / raster_width))
            raster_x = 0
        if raster_x + raster_width > self.x_size:
            window_width = int(window_width * (float(self.x_size - raster_x) / raster_width))
            raster_width = self.x_size - raster_x

        window_y = 0
        if raster_y < 0:
            y_shift = abs(raster_y)
            window_y = int(window_height * (float(y_shift) / raster_height))
            window_height = window_height - window_y
            raster_height = raster_height - int(raster_height * (float(y_shift) / raster_height))
            raster_y = 0
        if raster_y + raster_height > self.y_size:
            window_height = int(window_height * (float(self.y_size - raster_y) / raster_height))
            raster_height = self.y_size - raster_y

        return (raster_x, raster_y, raster_width, raster_height), (window_x, window_y, window_width, window_height)

    def resample_matrix(self, matrix, target_shape):
        """
        Resample the given matrix to the target shape.

        :param matrix: Input matrix.
        :param target_shape: Target shape (width, height).
        :return: Resampled matrix.
        """
        bands, raster_width, raster_height = matrix.shape
        window_width, window_height = target_shape
        zoom_factors = (1, window_width / raster_width, window_height / raster_height)
        resampled_matrix = zoom(matrix, zoom_factors, order=0)
        return resampled_matrix

    def generate_tiles(self,return_part_data=False):
        """
        Generate tiles for the given data.

        :return: List of tiles.
        """
        tile_list = []
        zoom_level = self.max_zoom
        query_size = self.tile_size
        delta_z = zoom_level - self.min_zoom

        for tile_y in range(self.max_tile_y, self.min_tile_y - 1, -1):
            for tile_x in range(self.min_tile_x, self.max_tile_x + 1):
                try:
                    final_tile_y = self.get_y_tile(tile_y, zoom_level)

                    bounds = self.mercator.TileBounds(tile_x, tile_y, zoom_level)
                    raster_bounds, window_bounds = self.geo_query(bounds, query_size=query_size)

                    raster_x, raster_y, raster_width, raster_height = raster_bounds
                    window_x, window_y, window_width, window_height = window_bounds
                    
                    sub_data = self.data[:, raster_y:raster_y + raster_height, raster_x:raster_x + raster_width]

                    target_data = np.zeros((sub_data.shape[0], self.tile_size, self.tile_size))
                    target_mask = np.zeros((self.tile_size, self.tile_size), dtype=bool)

                    part_data = self.resample_matrix(sub_data, (window_height, window_width))
                    target_data[:, window_y:window_y + window_height, window_x:window_x + window_width] = part_data
                    target_mask[window_y:window_y + window_height, window_x:window_x + window_width] = True

                    image_array = np.transpose(target_data, (1, 2, 0))
                    alpha_channel = target_mask.astype(np.uint8) 
                    rgba_array = np.dstack((image_array, alpha_channel))

                    if return_part_data:
                        tile_list.append({
                            'data': part_data,
                            'current_index': f'{zoom_level}/{tile_x}/{final_tile_y}',
                            'current_level': zoom_level,
                            'min_index': f'{self.min_zoom}/{tile_x // (2**delta_z)}/{final_tile_y // (2**delta_z)}',
                            'min_level': self.min_zoom,
                            'window_y':window_y,
                            'window_x':window_x,
                            'window_height':window_height,
                            'window_width':window_width
                        })
                    else:
                        tile_list.append({
                            'data': rgba_array,
                            'current_index': f'{zoom_level}/{tile_x}/{final_tile_y}',
                            'current_level': zoom_level,
                            'min_index': f'{self.min_zoom}/{tile_x // (2**delta_z)}/{final_tile_y // (2**delta_z)}',
                            'min_level': self.min_zoom
                        })
                except:
                    continue
        
        return tile_list

class OverviewTileGenerator:
    def __init__(self, tile_list, nodata_value=0, tile_size=256):
        """
        Initialize the OverviewTileGenerator.

        :param tile_list: List of tiles to generate overviews from.
        :param nodata_value: No data value.
        :param tile_size: Size of each tile.
        :param min_level: Minimum zoom level.
        """
        self.current_tile_list = tile_list
        self.all_tile_dict = {}
        self.tile_size = tile_size

    # def get_next_index_list(self, current_index_list):
    #     """
    #     Get the next level tile indices from the current level indices.

    #     :param current_index_list: List of current level tile indices.
    #     :return: List of next level tile indices.
    #     """
    #     next_index_list = [self.get_next_index(current_index) for current_index in current_index_list]
    #     next_index_list = list(set(next_index_list))
    #     return next_index_list

    def parse_index(self, current_index):
        """
        Parse the tile index into zoom level, tile x, and tile y.

        :param current_index: Tile index string.
        :return: Tuple of (zoom_level, tile_x, tile_y).
        """
        parts = current_index.split('/')
        zoom_level = int(parts[0])
        tile_x = int(parts[1])
        tile_y = int(parts[2])
        return zoom_level, tile_x, tile_y

    def get_next_index(self, current_index):
        """
        Get the next level tile index from the current level index.

        :param current_index: Current level tile index.
        :return: Next level tile index.
        """
        zoom_level, tile_x, tile_y = self.parse_index(current_index)
        next_zoom_level = zoom_level - 1
        next_tile_x = tile_x // 2
        next_tile_y = tile_y // 2
        return f'{next_zoom_level}/{next_tile_x}/{next_tile_y}'
    
    def resample_matrix(self, matrix, target_shape):
        """
        Resample the given matrix to the target shape.

        :param matrix: Input matrix.
        :param target_shape: Target shape (width, height).
        :return: Resampled matrix.
        """
        raster_width, raster_height, bands= matrix.shape
        window_width, window_height = target_shape
        zoom_factors = (window_width / raster_width, window_height / raster_height, 1)
        resampled_matrix = zoom(matrix, zoom_factors, order=0)
        return resampled_matrix

    def generate_next_tiles(self):
        """
        Generate overview tiles for the given tiles list to parent level.

        :return: List of overview tiles.
        """
        next_tile = []
        current_tile_list = self.current_tile_list
        current_index = current_tile_list[0]['current_index']
        band_size = current_tile_list[0]['data'].shape[-1]
        next_index = self.get_next_index(current_index)
        current_tile_list = [tile for tile in current_tile_list if tile['min_index']==next_index]

        zoom_level, tile_x, tile_y = self.parse_index(next_index)
        merge_data = np.zeros((2 * self.tile_size, 2 * self.tile_size, band_size))
        for tile in current_tile_list:
            current_index = tile['current_index']
            _,x,y = self.parse_index(current_index)
            merge_data[(y - 2 * tile_y) * self.tile_size:((y - 2 * tile_y) * self.tile_size + self.tile_size),
                    (x - 2 * tile_x) * self.tile_size:((x - 2 * tile_x) * self.tile_size + self.tile_size), :] = tile['data']
        next_data = self.resample_matrix(merge_data, (self.tile_size, self.tile_size))
        next_tile = {
            'data': next_data,
            'current_index': f'{zoom_level}/{tile_x}/{tile_y}',
            'current_level': zoom_level,
            'min_index': f'{zoom_level-1}/{tile_x // 2}/{tile_y // 2}',
            'min_level': zoom_level-1
        }
        return next_tile
    
def alter_min_index(result,min_level):
    """
        result: {
                    'current_index': f'{zoom_level}/{tile_x}/{final_tile_y}',
                    'current_level': zoom_level,
                    'min_index': f'{self.min_zoom}/{tile_x // (2**delta_z)}/{final_tile_y // (2**delta_z)}',
                    'min_level': self.min_zoom
                }
    """
    current_index = result['current_index']
    result['min_index'] = get_min_index(current_index,min_level=min_level)
    result['min_level'] = min_level
    return result

def get_min_index(current_index,min_level):
    zoom_level, tile_x, tile_y = parse_index(current_index)
    delta_z = zoom_level-min_level
    min_index = f'{min_level}/{tile_x // (2**delta_z)}/{tile_y // (2**delta_z)}'
    return min_index

def parse_index(current_index):
    """
    Parse the tile index into zoom level, tile x, and tile y.

    :param current_index: Tile index string.
    :return: Tuple of (zoom_level, tile_x, tile_y).
    """
    parts = current_index.split('/')
    zoom_level = int(parts[0])
    tile_x = int(parts[1])
    tile_y = int(parts[2])
    return zoom_level, tile_x, tile_y

def MergeTileRecords(tile_list, nodata_value=0):

    data_matrix_list = []
    alpha_list = []
    for tile in tile_list:
        _data_matrix = tile['data'][:, :, :-1]  
        _alpha = tile['data'][:, :, -1]        
        data_matrix_list.append(_data_matrix)
        alpha_list.append(_alpha)

    
    alpha_sum = np.sum(alpha_list, axis=0)
    alpha_sum[alpha_sum == 0] = 1  
    data_matrix = np.divide(np.sum(data_matrix_list, axis=0), alpha_sum[..., np.newaxis])
    alpha = np.max(alpha_list, axis=0)
    data = np.dstack((data_matrix, alpha))

    current_index = tile_list[0]['current_index']
    zoom_level = tile_list[0]['current_level']
    if 'min_index' in tile_list[0].keys():
        min_index = tile_list[0]['min_index']
    else:
        min_index = None
    if 'min_level' in tile_list[0].keys():
        min_level = tile_list[0]['min_level']
    else:
        min_level = None

    result = {
                'data': data,
                'current_index': current_index,
                'current_level': zoom_level,
                'min_index': min_index,
                'min_level': min_level
            }
    
    return result

def scale_to_uint8(band_data, min_val, max_val):
    """
    scale band_data to [0,255]
    """
    return np.interp(band_data, (min_val, max_val), (0, 255)).astype(np.uint8)

      
def save_matrix_as_png(rgba_array, output_path):
    """
    Save the given RGBA array as a PNG image.

    :param rgba_array: RGBA array.
    :param output_path: Output file path.
    """
    image = Image.fromarray(rgba_array.astype(np.uint8))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)