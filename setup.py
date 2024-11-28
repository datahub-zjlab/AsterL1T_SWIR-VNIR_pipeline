from setuptools import setup, find_packages

setup(
    name='aster_core',
    version='0.0.2',
    packages=find_packages(),
    install_requires=[
        'matplotlib==3.9.2',
        'numpy==2.1.2',
        'opencv_python==4.10.0.84',
        'pandas==2.2.3',
        'Pillow==11.0.0',
        'psycopg2==2.9.9',  
        'pyproj==3.6.1',
        'rasterio==1.3.10',
        'scipy==1.14.1',
        'Shapely==2.0.6',
        'scikit-image==0.24.0'
    ],
    entry_points={
        'console_scripts': [
        ],
    },
)