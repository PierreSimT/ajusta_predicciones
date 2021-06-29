import os
from osgeo import gdal, osr, ogr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import random
import rasterio
import glob

# CALCULA LA INCLINACION
def calculate_slope(DEM):
    gdal.DEMProcessing('assets/slope.tif', DEM, 'slope')
    with rasterio.open('assets/slope.tif') as dataset:
        slope=dataset.read(1)
    return slope

# CALCULA LA ORIENTACION
def calculate_aspect(DEM):
    gdal.DEMProcessing('assets/aspect.tif', DEM, 'aspect')
    with rasterio.open('assets/aspect.tif') as dataset:
        aspect=dataset.read(1)
    return aspect

# CALCULA LA TRANSFORMACION A 1km
def warp_to_1000():
    dataset = gdal.Open('assets/1000_pixel.tif')
    filepath = 'assets/1000_pixel.tif'

    return dataset, filepath

def world_to_pixel(geo_matrix, x, y):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate
    """
    ul_x= geo_matrix[0]
    ul_y = geo_matrix[3]
    x_dist = geo_matrix[1]
    y_dist = geo_matrix[5]
    pixel = int((x - ul_x) / x_dist)
    line = -int((ul_y - y) / y_dist)
    return pixel, line

# Transformacion de Wx y Wy a vv y dv
def transform_vars(data):
    """Funcion que transforma los outputs de Wx y Wy a VV y DV

    Args:
        data (DataFrame): DataFrame que contiene solo Wx y Wy

    Returns:
        DataFrame: DataFrame con los resultados pasados a VV y DV
    """
    df_trans = pd.DataFrame(data, columns=['Wx', 'Wy'])

    df_trans['vv'] = np.sqrt(np.square(df_trans['Wx']) + np.square(df_trans['Wy']))
    df_trans['dv'] = np.arctan2(df_trans['Wy'], df_trans['Wx']) * (180 / np.pi) + 180

    for index, row in df_trans.iterrows():
        row['dv'] = (row['dv'] + 180) % 360

    return df_trans[['dv', 'vv']]

def get_random_station(data, num):
    """Funcion que obtiene un numero de estaciones aleatorias

    Args:
        data (DataFrame): DataFrame donde se encuentran los datos de las estaciones
        num (Integer): Numero de estaciones a devolver

    Returns:
        [DataFrame, DataFrame]: Array de DataFrames, en el primero se encuentra los datos sin las estaciones y el segundo las estaciones sacadas.
    """

    result_data = data.copy()
    stations = result_data['name'].unique().tolist()
    random_stations = pd.DataFrame()

    for i in range(0, num):
        position = random.randint(0, len(stations) - 1)

        station_to_get = stations.pop(position)

        random_stations = random_stations.append(
            result_data[result_data['name'] == station_to_get])

        result_data = result_data.drop(
            result_data[result_data['name'] == station_to_get].index)

    return result_data, random_stations