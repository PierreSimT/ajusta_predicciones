import os

import logging
from numpy.core import shape_base

import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

from ModelSci import Model


class RandomForest(Model):
    """Class of model RandomForestRegressor

    Args:
        Model (class): Parent class of all models
    """

    name = "RandomForest"

    def train(self, X_train, X_test, y_train, y_test, output_directory, n_epochs):
        """Entrenamiento del modelo

        Args:
            X_train (DataFrame): Conjunto de entrenamiento
            X_test (DataFrame): Conjunto de validacion
            y_train (DataFrame): Resultados del conjunto de entrenamiento
            y_test (DataFrame): Resultados del conjunto de validacion
            n_epochs (int): Numero de epochs en caso de entrenar una red neuronal
        """

        self.model = RandomForestRegressor(n_jobs=-1)

        X_train_t = self.scaler.fit_transform(X_train)
        X_test_t = self.scaler.transform(X_test)

        logging.info("RandomForestRegressor: Training Model")
        self.model.fit(X_train_t, y_train)
        logging.info("RandomForestRegressor: Finished Training Model")

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        path = output_directory + '/' + self.name

        try:
            os.mkdir(path)
        except OSError as error:
            print(error)

        plt.figure(figsize=(20, 8))
        plt.title("Feature Importances")
        plt.bar(range(X_train.shape[1]),
                importances[indices], color="r", align="center")
        plt.xticks(range(X_train.shape[1]),
                   X_train.columns[indices], rotation=90)
        plt.xlim([-1, X_train.shape[1]])
        plt.savefig(path + '/features_importances.png')
        plt.close()

        # explainer = shap.TreeExplainer(self.model)
        # shap_values = explainer.shap_values(X_test_t)
        # shap.summary_plot(shap_values, X_test_t, feature_names=X_test.columns, max_display=50, plot_size=(16, 20), show=False)
        # plt.savefig(path + '/feature_importances_shap.png')
        # plt.close()

        pd.DataFrame([self.model.get_params()]).to_csv(f"{path}/best_params.csv")

    def save_model(self, pathname):
        pkl_filename = pathname + '.pkl'

        with open(pkl_filename, 'wb') as file:
            pickle.dump(self.model, file)
