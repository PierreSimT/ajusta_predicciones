import datetime
import numpy as np
import pandas as pd
import pyproj

from osgeo import gdal, osr, ogr

from modules.Common import calculate_aspect, calculate_slope, warp_to_1000, world_to_pixel


class DataPreprocessor:

    def __init__(self, data):
        self.df = data.copy()

    def get_data(self):
        return self.df

    def transform_geo(self):

        df = self.df.copy()

        # INTRODUCIMOS INCLINACION Y ORIENTACION
        dataset, filepath = warp_to_1000()
        slope = calculate_slope(filepath)
        aspect = calculate_aspect(filepath)

        all_stations = df['name'].unique()
        grouped_data = df.groupby(
            ['name', 'latitude', 'longitude']).size().reset_index()

        ds = dataset
        target = osr.SpatialReference(wkt=ds.GetProjection())
        source = osr.SpatialReference()
        source.ImportFromEPSG(4326)
        transform = osr.CoordinateTransformation(source, target)

        for station in all_stations:

            latitude = grouped_data[grouped_data['name']
                                    == station]['latitude'].values[0]
            longitude = grouped_data[grouped_data['name']
                                        == station]['longitude'].values[0]

            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(latitude, longitude)
            point.Transform(transform)

            x, y = world_to_pixel(
                ds.GetGeoTransform(), point.GetX(), point.GetY())

            df.loc[df[df['name'] == station].index,
                    'inclination'] = np.radians(slope[y][x])
            df.loc[df[df['name'] == station].index,
                    'orientation'] = np.radians(aspect[y][x])

        df['inclination Sin'] = np.sin(df['inclination'])
        df['inclination Cos'] = np.cos(df['inclination'])
        df['orientation Sin'] = np.sin(df['orientation'])
        df['orientation Cos'] = np.cos(df['orientation'])

        # TRANSFORMAMOS LA POSICION GEOGRAFICA A UN ESPACIO 2D
        eqc = pyproj.Proj(proj='eqc', ellps='WGS84', datum='WGS84')
        lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

        transformer = pyproj.Transformer.from_proj(lla, eqc)

        df['longitude'] = df['longitude'] * (np.pi / 180)
        df['latitude'] = df['latitude'] * (np.pi / 180)

        df['X'], df['Y'], df['Z'] = transformer.transform(
            xx=df['longitude'], yy=df['latitude'], zz=df['elevation'], radians=True) # df['Z']

        # SACAMOS DATOS NO NECESARIOS
        df = df.drop(['latitude', 'longitude', 'elevation',
                      'inclination', 'orientation'], axis=1)

        self.df = df.copy()

    def transform_date(self):

        df = self.df.copy()

        # ORDENAMOS LOS DATOS POR FECHA
        df = df.sort_values('Datetime')

        # TRANSFORMAMOS LAS FECHAS
        date_time = pd.to_datetime(
            df['Datetime'], format='%Y-%m-%d %H:%M:%S')

        timestamp_s = date_time.map(datetime.datetime.timestamp)

        day = 24*60*60
        year = (365.2425)*day

        df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

        self.df = df.copy()

    def transform_wind_data(self, num_of_sims):

        df = self.df.copy()

        wv_2 = df.pop('vv|m/s')  #  vv|m/s
        wd_rad_2 = df.pop('dv|grados') * np.pi / 180  #  dv|grados

        zero_order_wv = df.pop('Zero_Order_Hold_vv')
        zero_order_wd_rad = df.pop('Zero_Order_Hold_dv') * np.pi / 180

        df['Zero_Order_Wx'] = zero_order_wv * np.cos(zero_order_wd_rad)
        df['Zero_Order_Wy'] = zero_order_wv * np.sin(zero_order_wd_rad)

        df['Wx'] = wv_2 * np.cos(wd_rad_2)
        df['Wy'] = wv_2 * np.sin(wd_rad_2)

        # CAMBIAR RANGO DEPENDIENDO DE simulDay1 o simulDay2
        for i in range(0, num_of_sims):  # 45
            wv = df.pop('Sim_vv_' + str(i))

            # Convert to radians.
            wd_rad = df.pop('Sim_dv_' + str(i)) * np.pi / 180

            # Calculate the wind x and y components.
            df['Wx_' + str(i)] = wv*np.cos(wd_rad)
            df['Wy_' + str(i)] = wv*np.sin(wd_rad)

        self.df = df.copy()
