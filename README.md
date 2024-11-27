<!-- TODO --><!-- TODO -->
ASTER L1T remote sence VNIR and SWIR data preprocess, mosaic and functional group calculation for geoscience.
### **Purpose**

This project is to process Aster L1t VNIR and SWIR dataset.

### **Get Started**

1. Prepare enviroments  

`pip install -r requirements.txt`

2. Install packages
   
`pip install -e path/to/your/package`


### **Method**
1. `aster_core.GlobalGrid`
    
    (1) The `GlobalRasterGrid` class defines a raster grid with specified resolution, reference coordinate system (EPSG:3857), and bounding box, and provides methods to calculate grid bounds, tile counts, and tile indices.
    
    (2) It supports generating tile bounding boxes, polygons, and geotransforms based on tile indices, and can identify intersecting tiles for given bounds or points.
    
    (3) The class also allows for the creation of different raster grids with varying resolutions and tile sizes, such as MODIS and ASTER grids, demonstrating its flexibility in handling various raster data configurations.

2. `aster_core.preprocess`
    
    (1) `cal_radiance`

    (2) `cal_toa`

3. `aster_core.atmospheric_correction`

    6s model based atmospheric correction with LUT and interpolation.

4. `aster_core.color_transfer`

    Modis reference based color tranfer to lower difference between original granules.

5. `aster_core.functioan_group`

    'ferric_iron', 'ferrous_iron', 'laterite', 'gossan', 'ferrous_silicates', 'ferric_oxides',
    'carbonate_chlorite_epidote', 'mg_oh_alteration', 'amphibole_mgoh', 'amphibole', 'dolomite', 
    'sericite_muscovite_illite_smectite', 'alunite_kaolinite_pyrophyllite', 
    'phengitic', 'muscovite', 'kaolinite', 'clay', 'kaolinite_argillic', 'alunite_advanced_argillic', 
    'al_oh_alteration', 'calcite', 'ndwi', 'ndvi'

### **Related Material**
https://pubs.usgs.gov/sir/2010/5090/o/

https://lpdaac.usgs.gov/products/ast_l1tv003/

### **Contact us**
[Chen Ziyang ](chenzy@zhejianglab.org), Zhejiang Lab, CHINA




