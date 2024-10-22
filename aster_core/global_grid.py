import numpy as np
from shapely import Point
from rasterio.coords import BoundingBox
from rasterio.transform import from_bounds
from aster_core.utils import bbox2bbox,bbox2polygon,affine_to_geotransform
import math

class GlobalRasterGrid:
    '''
    The raster has the following characteristics:
    1. It has a specified resolution.
    2. It is based on a specified geographic reference coordinate system (currently using EPSG:3857).
    3. It has a specified bounding box.
    4. A tile class is defined based on the specified tile size.
    5. The tile class can return the bounding box of a tile based on the tile index.
    6. The tile class can return all tiles that cover a specified bounding box.

    A breif introduction of epsg:3857:
    | EPSG:3857 is a coordinate system used for web map projections, also known as Web Mercator. 
    | It is based on the Mercator projection but differs from the traditional Mercator projection (such as EPSG:4326) in that EPSG:3857 uses meters as the unit of measurement instead of degrees. 
    | This makes it very popular in web mapping services because it provides consistent zoom levels and performance.
    '''
    def __init__(self, resolution=30, tile_size=256):
        # Pixel scale in x and y directions, usually x_res is postive and y_res is negative
        self.res_x = resolution
        self.res_y = -resolution
        
        # Tile size of raster
        self.tile_size = tile_size

        # Origin shift calculation
        self.projection = 'epsg:3857'
        self.originShift = 2 * math.pi * 6378137 / 2.0

        # Calculate the left, right, top, and bottom coordinates of the grid
        self.left = self._calculate_left()
        self.right = self._calculate_right()
        self.top = self._calculate_top()
        self.bottom = self._calculate_bottom()

    def _calculate_left(self):
        # Calculate the leftmost coordinate of the grid
        return -np.abs(np.floor(self.originShift * 2 / np.abs(self.res_x)) * np.abs(self.res_x) / 2)

    def _calculate_right(self):
        # Calculate the rightmost coordinate of the grid
        return np.abs(np.floor(self.originShift * 2 / np.abs(self.res_x)) * np.abs(self.res_x) / 2)

    def _calculate_top(self):
        # Calculate the topmost coordinate of the grid
        return np.abs(np.floor(self.originShift * 2 / np.abs(self.res_y)) * np.abs(self.res_y) / 2)

    def _calculate_bottom(self):
        # Calculate the bottommost coordinate of the grid
        return -np.abs(np.floor(self.originShift * 2 / np.abs(self.res_y)) * np.abs(self.res_y) / 2)

    def get_bounds(self):
        # Get the bounding box of the grid
        return BoundingBox(left=self.left, bottom=self.bottom, right=self.right, top=self.top)

    def get_tile_count(self):
        # Get the number of tiles in the x and y directions
        num_tiles_x = np.floor((self.right - self.left) / (self.res_x*self.tile_size)) # from left to right
        num_tiles_y = np.floor((self.bottom - self.top) / (self.res_y*self.tile_size)) # from top to bottom
        return int(num_tiles_x), int(num_tiles_y)

    def get_tile_geotransform(self,*args,affine_flag=True):
        tile_bbox = self.get_tile_bounds(*args)
        affine = from_bounds(*tile_bbox, self.tile_size, self.tile_size)
        geotransform = affine_to_geotransform(affine)
        if affine_flag:
            return affine
        else:
            return geotransform

    def get_tile_bounds(self, *args):
        # Check if the input is a single tuple or two separate arguments
        if len(args) == 1 and isinstance(args[0], tuple) and len(args[0]) == 2:
            tile_x, tile_y = args[0]
        elif len(args) == 2:
            tile_x, tile_y = args
        else:
            raise ValueError("Input must be either a tuple (tile_x, tile_y) or two separate arguments tile_x and tile_y")

        # Get the bounding box of a specific tile
        left = self.left + tile_x * self.res_x * self.tile_size
        right = left + self.res_x * self.tile_size
        top = self.top + tile_y * self.res_y * self.tile_size
        bottom = top + self.res_y * self.tile_size
        return BoundingBox(left=left, bottom=bottom, right=right, top=top)
    
    def get_tile_polygon(self,*args,output_crs=None):
        tile_bbox = self.get_tile_bounds(*args)
        if not output_crs is None:
            tile_bbox = bbox2bbox(tile_bbox,self.projection,output_crs)
        tile_polygon = bbox2polygon(tile_bbox)
        return tile_polygon

    def get_tile_index(self, input):
        if isinstance(input, tuple):
            if len(input) == 2:
                x, y = input
                bounds = BoundingBox(left=x,right=x,top=y,bottom=y)
            elif len(input) == 4:
                bounds = input
            else:
                raise ValueError("Tuple must contain exactly two elements (x, y).")
        elif isinstance(input, Point):
            x, y = input.x, input.y
            bounds = BoundingBox(left=x,right=x,top=y,bottom=y)
        # elif isinstance(input, (int, float)):
        #     x, y = input, input
        #     bounds = BoundingBox(left=x,right=x,top=y,bottom=y)
        else: 
            raise ValueError("Unsupported input type.")

        # Calculate the tile indices that intersect with the given bounds
        left_offset = (bounds.left - self.left) / (self.res_x*self.tile_size)
        right_offset = (bounds.right - self.left) / (self.res_x*self.tile_size)
        upper_offset = (bounds.top - self.top) / (self.res_y*self.tile_size)
        lower_offset = (bounds.bottom - self.top) / (self.res_y*self.tile_size)

        # Calculate the range of tile indices that intersect with the bounds
        min_x = max(math.floor(left_offset), 0)
        max_x = min(math.floor(right_offset), self.get_tile_count()[0])
        min_y = max(math.floor(upper_offset), 0)
        max_y = min(math.floor(lower_offset), self.get_tile_count()[1])

        return min_x, max_x, min_y, max_y
    
    def get_tile_list(self, input):
        min_x, max_x, min_y, max_y = self.get_tile_index(input)
        coordinates = [(x, y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)]
        return coordinates

    def get_grided_bounds(self, bounds):
        # Get the bounding box of the grid that intersects with the given bounds
        min_x, max_x, min_y, max_y = self.get_tile_index(bounds)
        left = self.left + min_x * (self.res_x*self.tile_size)
        right = self.left + (max_x + 1) * (self.res_x*self.tile_size)
        bottom = self.top + (max_y + 1) * (self.res_y*self.tile_size)
        top = self.top + min_y * (self.res_y*self.tile_size)
        return BoundingBox(left=left, bottom=bottom, right=right, top=top)
    
    def get_index(self):
        return self.get_tile_index(self.get_bounds())

modis_resolution = 500
modis_tile_size = 256
modis_global_grid = GlobalRasterGrid(resolution=modis_resolution, tile_size=modis_tile_size)

aster_resolution = 30
aster_tile_size = 1024
aster_global_grid = GlobalRasterGrid(resolution=aster_resolution, tile_size=aster_tile_size)