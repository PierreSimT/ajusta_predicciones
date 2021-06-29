from abc import ABC

import os

import logging
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, mean_bias_error
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

class Model(ABC):

    name = "TemplateClass"

    def __init__(self, data, n_outputs):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.model = None

    def train(self, X_train, X_test, y_train, y_test, output_directory, n_epochs):
        pass

    def get_statistics(self, X_test, y_test, time, output_directory):

        logging.info(f'[{self.name}][Stats]: Starting')

        path = output_directory

        try:
            result = pd.read_csv(path + '/statistics.csv')
        except:
            print("Couldnt read stats file. Creating...")
            result = pd.DataFrame(columns=['Model', 'Variable', 'MAE', 'MBE', 'MSE', 'MSRE', 'EVS', 'CORR', 'STD', 'Time'])

        X_test = self.scaler.transform(X_test)
        
        predictions = self.model.predict(X_test)

        logging.info(f'[{self.name}][Stats]: Running')

        for column in range(0, len(y_test.columns)):

            mae = mean_absolute_error(y_test.values[:, column], predictions[:, column])
            mbe = mean_bias_error(y_test.values[:, column], predictions[:, column])
            mse = mean_squared_error(y_test.values[:, column], predictions[:, column])
            msre = np.sqrt(mean_squared_error(y_test.values[:, column], predictions[:, column]))
            evs = explained_variance_score(y_test.values[:, column], predictions[:, column])
            std = np.std([y_test.values[:, column], predictions[:, column]])
            corr = np.corrcoef(y_test.values[:, column], predictions[:, column])[0,1]

            result = result.append({'Model': self.name, 'Variable': str(y_test.columns[column]), 'MAE': mae, 'MBE': mbe, 'MSE': mse, 'MSRE': msre, 'EVS': evs, 'CORR': corr, 'STD': std, 'Time': time}, ignore_index=True)
            result.to_csv(path + '/statistics.csv', index=False, index_label=False)

        logging.info(f'[{self.name}][Stats]: Done')

    def get_monthly_statistics(self, data, drop_columns, output_column, output_directory):

        logging.info(f'[{self.name}][Stats][Month]: Starting')

        result = pd.DataFrame(columns=['Month', 'Variable', 'MAE', 'MBE', 'MSE', 'MSRE', 'EVS', 'CORR', 'STD'])
        df = data.copy()

        path = output_directory + '/' + self.name

        try:
            os.mkdir(path)
        except OSError as error:
            print(error)

        logging.info('MonthlyStats: Generating months')
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')
        df['month'] = df['Datetime'].map(lambda x: x.strftime('%B'))
        
        df = df.drop(drop_columns, axis=1)
        months = df['month'].unique()

        logging.info(f'[{self.name}][Stats][Month]: Running')
        for month in months:

            preddf = df[df['month'] == month].drop('month', axis=1)

            X = self.scaler.transform(preddf.drop(output_column, axis=1))
            y = preddf[output_column]

            predictions = self.model.predict(X)

            for column in range(0, len(y.columns)):

                mae = mean_absolute_error(y.values[:, column], predictions[:, column])
                mbe = mean_bias_error(y.values[:, column], predictions[:, column])
                mse = mean_squared_error(y.values[:, column], predictions[:, column])
                msre = np.sqrt(mean_squared_error(y.values[:, column], predictions[:, column]))
                evs = explained_variance_score(y.values[:, column], predictions[:, column])
                std = np.std([y.values[:, column], predictions[:, column]])
                corr = np.corrcoef(y.values[:, column], predictions[:, column])[0,1]

                result = result.append({'Month': month, 'Variable': str(y.columns[column]), 'MAE': mae, 'MBE': mbe, 'MSE': mse, 'MSRE': msre, 'EVS': evs, 'CORR': corr, 'STD': std}, ignore_index=True)
                result.to_csv(path + '/monthly.csv')

            logging.info(f'[{self.name}][Stats][Month]: Saving {path}')

        logging.info(f'[{self.name}][Stats][Month]: Done')


    def get_hourly_statistics(self, data, drop_columns, output_column, output_directory):

        logging.info(f'[{self.name}][Stats][Hour]: Starting')
        result = pd.DataFrame(columns=['Hour', 'Variable', 'MAE', 'MBE', 'MSE', 'MSRE', 'EVS', 'CORR', 'STD'])
        df = data.copy()

        path = output_directory + '/' + self.name

        try:
            os.mkdir(path)
        except OSError as error:
            print(error)

        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')
        df['hour'] = df['Datetime'].map(lambda x: x.strftime('%H:%M'))
        
        df = df.drop(drop_columns, axis=1)
        hours = df['hour'].unique()

        logging.info(f'[{self.name}][Stats][Hour]: Running')
        for hour in hours:

            preddf = df[df['hour'] == hour].drop('hour', axis=1)

            X = self.scaler.transform(preddf.drop(output_column, axis=1))
            y = preddf[output_column]

            predictions = self.model.predict(X)

            for column in range(0, len(y.columns)):

                mae = mean_absolute_error(y.values[:, column], predictions[:, column])
                mbe = mean_bias_error(y.values[:, column], predictions[:, column])
                mse = mean_squared_error(y.values[:, column], predictions[:, column])
                msre = np.sqrt(mean_squared_error(y.values[:, column], predictions[:, column]))
                evs = explained_variance_score(y.values[:, column], predictions[:, column])
                std = np.std([y.values[:, column], predictions[:, column]])
                corr = np.corrcoef(y.values[:, column], predictions[:, column])[0,1]
                

                result = result.append({'Hour': hour, 'Variable': str(y.columns[column]), 'MAE': mae, 'MBE': mbe, 'MSE': mse, 'MSRE': msre, 'EVS': evs, 'CORR': corr, 'STD': std}, ignore_index=True)
                result.to_csv(path + '/hourly.csv')

            logging.info(f'[{self.name}][Stats][Hour]: Saving {path}')

        logging.info(f'[{self.name}][Stats][Hour]: Done')

    def get_hour_monthly_statistics(self, data, drop_columns, output_column, output_directory):

        logging.info(f'[{self.name}][Stats][Month][Hour]: Starting')
        result = pd.DataFrame(columns=['Month', 'Variable', 'Hour', 'MAE', 'MBE', 'MSE', 'MSRE', 'EVS', 'CORR', 'STD'])
        df = data.copy()

        path = output_directory + '/' + self.name

        try:
            os.mkdir(path)
        except OSError as error:
            print(error)

        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')
        df['month'] = df['Datetime'].map(lambda x: x.strftime('%B'))
        df['hour'] = df['Datetime'].map(lambda x: x.strftime('%H:%M'))
        
        df = df.drop(drop_columns, axis=1)

        months = df['month'].unique()
        hours = df['hour'].unique()

        logging.info(f'[{self.name}][Stats][Month][Hour]: Running')
        for month in months:
            
            monthdf = df[df['month'] == month]

            for hour in hours:

                preddf = monthdf[monthdf['hour'] == hour].drop(['month', 'hour'], axis=1)

                X = self.scaler.transform(preddf.drop(output_column, axis=1))
                y = preddf[output_column]

                predictions = self.model.predict(X)

                for column in range(0, len(y.columns)):

                    mae = mean_absolute_error(y.values[:, column], predictions[:, column])
                    mbe = mean_bias_error(y.values[:, column], predictions[:, column])
                    mse = mean_squared_error(y.values[:, column], predictions[:, column])
                    msre = np.sqrt(mean_squared_error(y.values[:, column], predictions[:, column]))
                    evs = explained_variance_score(y.values[:, column], predictions[:, column])
                    std = np.std([y.values[:, column], predictions[:, column]])
                    corr = np.corrcoef(y.values[:, column], predictions[:, column])[0,1]

                    result = result.append({'Month': month, 'Variable': str(y.columns[column]), 'Hour': hour, 'MAE': mae, 'MBE': mbe, 'MSE': mse, 'MSRE': msre, 'EVS': evs, 'CORR': corr, 'STD': std}, ignore_index=True)
                    result.to_csv(path + '/monthly_hourly.csv')

                logging.info(f'[{self.name}][Stats][Month][Hour]: Saving {path}')
                
                
        for column in output_column:

            aresult = result[result["Variable"] == column].pivot("Month", "Hour", "MSRE")
            aresult.index = pd.to_datetime(aresult.index, format="%B")
            aresult = aresult.sort_index()
            aresult = aresult.rename(index=lambda x: x.strftime('%B'))

            plt.figure(figsize=(25,12))
            plt.title(self.name + ' - MSRE')
            sns.heatmap(aresult, linewidths=.25, annot=True, fmt='g')
            plt.savefig(path + '/' + column.split('|')[0] + '_hour_monthly.png')
            plt.close()

        logging.info(f'[{self.name}][Stats][Month][Hour]: Done')

    def get_station_statistics(self, data, drop_columns, output_column, output_directory):

        logging.info(f'[{self.name}][Stats][Station]: Starting')
        result = pd.DataFrame(columns=['Station', 'Variable', 'MAE', 'MBE', 'MSE', 'MSRE', 'EVS', 'CORR', 'STD'])
        df = data.copy()

        path = output_directory + '/' + self.name

        try:
            os.mkdir(path)
        except OSError as error:
            print(error)
        
        df['station'] = df['name']

        df = df.drop(drop_columns, axis=1)
        stations = df['station'].unique()

        for station in stations:

            preddf = df[df['station'] == station].drop('station', axis=1)

            X = self.scaler.transform(preddf.drop(output_column, axis=1))
            y = preddf[output_column]

            predictions = self.model.predict(X)

            for column in range(0, len(y.columns)):

                mae = mean_absolute_error(y.values[:, column], predictions[:, column])
                mbe = mean_bias_error(y.values[:, column], predictions[:, column])
                mse = mean_squared_error(y.values[:, column], predictions[:, column])
                msre = np.sqrt(mean_squared_error(y.values[:, column], predictions[:, column]))
                evs = explained_variance_score(y.values[:, column], predictions[:, column])
                std = np.std([y.values[:, column], predictions[:, column]])
                corr = np.corrcoef(y.values[:, column], predictions[:, column])[0,1]

            result = result.append({'Station': station, 'Variable': str(y.columns[column]), 'MAE': mae, 'MBE': mbe, 'MSE': mse, 'MSRE': msre, 'EVS': evs, 'CORR': corr, 'STD': std}, ignore_index=True)
            result.to_csv(path + '/station.csv')
    
        logging.info(f'[{self.name}][Stats][Station]: Done')
            

    def get_monthly_station_statistics(self, data, drop_columns, output_column, output_directory):

        logging.info(f'[{self.name}][Month][Station]: Starting')
        result = pd.DataFrame(columns=['Month', 'Variable', 'Station', 'MAE', 'MBE', 'MSE', 'MSRE', 'EVS', 'CORR', 'STD'])
        df = data.copy()

        path = output_directory + '/' + self.name

        try:
            os.mkdir(path)
        except OSError as error:
            print(error)
        
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')
        df['month'] = df['Datetime'].map(lambda x: x.strftime('%B'))
        df['station'] = df['name']

        df = df.drop(drop_columns, axis=1)
        months = df['month'].unique()
        stations = df['station'].unique()

        for station in stations:

            monthdf = df[df['station'] == station]

            for month in months:

                preddf = monthdf[monthdf['month'] == month].drop(['month', 'station'], axis=1)

                if preddf.empty:
                    mae = np.nan
                    mse = np.nan
                    msre = np.nan
                    evs = np.nan
                else:
                    # print(preddf)

                    X = self.scaler.transform(preddf.drop(output_column, axis=1))
                    y = preddf[output_column]

                    predictions = self.model.predict(X)

                    for column in range(0, len(y.columns)):

                        mae = mean_absolute_error(y.values[:, column], predictions[:, column])
                        mbe = mean_bias_error(y.values[:, column], predictions[:, column])
                        mse = mean_squared_error(y.values[:, column], predictions[:, column])
                        msre = np.sqrt(mean_squared_error(y.values[:, column], predictions[:, column]))
                        evs = explained_variance_score(y.values[:, column], predictions[:, column])
                        std = np.std([y.values[:, column], predictions[:, column]])
                        corr = np.corrcoef(y.values[:, column], predictions[:, column])[0,1]

                        result = result.append({'Month': month, 'Variable': str(y.columns[column]), 'Station': station, 'MAE': mae, 'MBE': mbe, 'MSE': mse, 'MSRE': msre, 'EVS': evs, 'CORR': corr, 'STD': std}, ignore_index=True)
                        result.to_csv(path + '/monhtly_station.csv')
        
        for column in output_column:

            aresult = result[result["Variable"] == column].pivot("Month", "Station", "MSRE")
            aresult.index = pd.to_datetime(aresult.index, format="%B")
            aresult = aresult.sort_index()
            aresult = aresult.rename(index=lambda x: x.strftime('%B'))

            plt.figure(figsize=(25,12))
            plt.title(self.name + ' - MSRE')
            sns.heatmap(aresult)
            plt.savefig(path + '/' + column.split('|')[0] + '_monthly_station.png')
            plt.close()

        logging.info(f'[{self.name}][Month][Station]: Done')