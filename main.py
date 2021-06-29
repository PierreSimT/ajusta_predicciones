import os 
import random
import logging
import argparse
import pandas as pd
import numpy as np
import geopandas as gpd

import time

import matplotlib.pyplot as plt

from shapely.geometry import Point

from modules.DataPreprocessor import DataPreprocessor
from modules.Common import get_random_station
from modules.TaylorDiagram import TaylorDiagram

from neural_network_tf import NeuralNetworkTF
from convolutional_network import ConvolutionalNetwork
from gradient_boosting_xgb import GradientBoostXGB
from random_forest import RandomForest
from mean import MeanModel

logging.basicConfig(filename='/home/pierresimt/work_dir/randomized_search/execution.log', encoding="utf-8", level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Test generalization of models")
parser.add_argument('input', type=open, help="Input train data file")
parser.add_argument('output', type=str, help="Output file with analysis")
parser.add_argument('n_output', type=int, help="Number of results of predictions")

parser.add_argument('-p', '--preprocess', action='store_true', help="Apply preprocessing to data")
parser.add_argument('-g', '--geo', action='store_true', help="Apply terrain preprocessing to data (-p must be added)")
parser.add_argument('-w', '--wind', type=int, help="Preprocess wind data with given number of simulations")

parser.add_argument('-d', '--drop', action='append', help="Columns to drop from input data")
parser.add_argument('-o', '--output_col', action='append', required=True, help="Column/s that will be used as outputs for the trained models")
parser.add_argument('-n', '--n_execution', type=int, default=1)
parser.add_argument('-s', '--n_stations', type=int, default=0)

args = parser.parse_args()

if __name__ == "__main__":

    logging.debug("[MAIN]: Starting...")
    logging.debug(f"[MAIN]: Training set: {args.input}")
    logging.debug(f"[MAIN]: Output : {args.output}")

    df = pd.read_csv(args.input, index_col=False)

    ## Posibilidad de transformar los datos, obtener inclinacion y orentacion
    if args.preprocess:
        #TRAIN DATASET
        logging.debug("[Transformer]: Transforming input data...")
        data_pre = DataPreprocessor(df)
        if args.wind:
            data_pre.transform_wind_data(args.wind)
        if args.geo:
            data_pre.transform_geo()
        data_pre.transform_date()
        df = data_pre.get_data()

        logging.debug("[Transformer]: Finished transforming input data...")

    orig_df = df.copy()

    if args.drop:
        df_d = df.drop(args.drop, axis=1)

    logging.info("[MAIN]: Creating validation and testing data for model")
    X = df_d.drop(args.output_col, axis=1)
    y = df_d[args.output_col]

    all_models = []

    all_models.append(MeanModel(df_d, args.n_output))
    all_models.append(GradientBoostXGB(df_d, args.n_output))
    all_models.append(RandomForest(df_d, args.n_output))
    all_models.append(ConvolutionalNetwork(df_d, args.n_output))
    all_models.append(NeuralNetworkTF(df_d, args.n_output))

    try:
        os.mkdir(f'./output/{args.output}')
    except:
        logging.debug(f"[Executor]: Couldn't create directory {args.output}")

    for ex in range(0, args.n_execution):
        
        if (args.n_stations != 0):
            pre_path = f'./output/{args.output}/{args.n_stations}'
        else:
            pre_path = f'./output/{args.output}/all'

        path = f'./{pre_path}/{ex}'

        try:
            os.mkdir(pre_path)
        except:
            logging.debug(f"[Executor]: Couldn't create directory {pre_path}")

        try:
            os.mkdir(path)
        except:
            logging.debug(f"[Executor]: Couldn't create directory {path}")

        if (args.n_stations == 0):

            df_aux = df.copy()
            df_train_validation = pd.DataFrame()

            # SACAR UNA DIA ALEATORIO DE CADA MES DE TODAS LAS ESTACIONES
            df_aux['Datetime'] = pd.to_datetime(df_aux['Datetime'], format='%Y-%m-%d %H:%M:%S')
            df_aux['month'] = df_aux['Datetime'].map(lambda x: x.strftime('%B'))
            df_aux['day'] = df_aux['Datetime'].map(lambda x: x.strftime('%d'))

            months = df_aux['month'].unique()

            for month in months:

                extract_df = df_aux[df_aux['month'] == month]
                days = extract_df['day'].unique()

                day_choice = random.choice(days)

                df_train_validation = df_train_validation.append(extract_df[extract_df['day'] == day_choice])

            df_train_train = df_aux[~df_aux.apply(tuple,1).isin(df_train_validation.apply(tuple,1))]

            df_train_validation = df_train_validation.drop(['month', 'day'], axis=1)
            df_train_train = df_train_train.drop(['month', 'day'], axis=1)

            df_test = df_train_validation.copy()

            df_train_train.to_csv(path + '/train.csv')
            df_train_validation.to_csv(path + '/test.csv')

            df_train_validation = df_train_validation.drop(args.drop, axis=1)
            df_train_train = df_train_train.drop(args.drop, axis=1)

            X_train = df_train_train.drop(args.output_col, axis=1)
            X_test = df_train_validation.drop(args.output_col, axis=1)

            y_train = df_train_train[args.output_col]
            y_test = df_train_validation[args.output_col]

        else:

            # SACAR CONJUNTO DE ESTACIONES
            df_test, df_train = get_random_station(df, args.n_stations)

            stations_used = df_train[['name']].drop_duplicates()
            stations_used.to_csv(f"{path}/stations_used.csv")
            
            df_train_aux = df_train.drop(args.drop, axis=1)
            X_train = df_train_aux.drop(args.output_col, axis=1)
            y_train = df_train_aux[args.output_col]

            df_test_aux = df_test.drop(args.drop, axis=1)
            X_test = df_test_aux.drop(args.output_col, axis=1)
            y_test = df_test_aux[args.output_col]

        for model in all_models:
            begin = time.time() # Inicio medidor tiempo

            model.train(X_train, X_test, y_train, y_test, path, 20)

            end = time.time() # Final medidor tiempo
            ex_time = end - begin

            model.get_statistics(X_test, y_test, ex_time, path)

            # model.save_model(model.name)

            model.get_hourly_statistics(df_test, args.drop, args.output_col, path)
            model.get_monthly_statistics(df_test, args.drop, args.output_col, path)
            model.get_hour_monthly_statistics(df_test, args.drop, args.output_col, path)
            model.get_station_statistics(df_test, args.drop, args.output_col, path)
            model.get_monthly_station_statistics(df_test, args.drop, args.output_col, path)

        ## Pintar Diagrama Taylor

        statistics = pd.read_csv(path + '/statistics.csv')
        statistics = statistics.drop([0])

        refStd = np.std(y_test.values)
        print(refStd)


        taylor = TaylorDiagram(refstd=refStd)
        fig = plt.figure(figsize=(11,8))
        ax2 = taylor.setup_axes(fig)

        min_value = np.min(statistics['MBE'].values)
        max_value = np.max(statistics['MBE'].values)

        t = fig.text(0.48, 0.93, "Taylor", horizontalalignment="center", fontsize=16)
        x=[0,0]
        y=[0,0]
        z=[min_value, max_value]
        plt.scatter(x,y,s=80, c=z, cmap= plt.cm.RdBu_r, marker=(1,0))

        for model in statistics['Model'].unique():
            
            std = statistics[statistics['Model'] == model]['STD'].values[0]
            mbe = statistics[statistics['Model'] == model]['MBE'].values[0]
            corr = statistics[statistics['Model'] == model]['CORR'].values[0]

            vals=[std, mbe, corr]
            col = plt.cm.RdBu_r((vals[1]-min_value)/(max_value-min_value),1)
            taylor.plot_sample(vals[2], vals[0], 'o', c= col, ms=10.0, label=f'{model}')

        plt.legend(bbox_to_anchor=(1.1, 1.1), loc='upper left', borderaxespad=0., prop={'size':10})
        cb = plt.colorbar(orientation='horizontal',shrink=0.6, format='%.1f')
        cb.set_label("MBE")

        plt.savefig(f"{path}/taylor.png", dpi = (200))
        plt.close()
