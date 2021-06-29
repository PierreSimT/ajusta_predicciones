import os
import logging
import pickle

import numpy as np
import pandas as pd

import shap
import xgboost as xgb

import matplotlib.pyplot as plt

from ModelSci import Model

class GradientBoostXGB(Model):
    """Clase del Modelo GradientBoostRegressor

    Args:
        Model (class): Clase padre de los modelos
    """

    name = "GradientBoosting"

    def train(self, X_train, X_test, y_train, y_test, output_directory, n_epochs):
        """Entrenamiento del modelo

        Args:
            X_train (DataFrame): Conjunto de entrenamiento
            X_test (DataFrame): Conjunto de validacion
            y_train (DataFrame): Resultados del conjunto de entrenamiento
            y_test (DataFrame): Resultados del conjunto de validacion
            n_epochs (int): Numero de epochs en caso de entrenar una red neuronal
        """

        self.model = xgb.XGBRegressor()

        X_train_t = self.scaler.fit_transform(X_train)
        X_test_t = self.scaler.transform(X_test)

        evalset = [(X_train_t, y_train), (X_test_t, y_test)]

        logging.info("GradientBoostingRegressor: Training Model")
        self.model.fit(X_train_t, y_train, eval_set=evalset, eval_metric='mae')
        logging.info("GradientBoostingRegressor: Finished Training Model")


        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        path = output_directory + '/' + self.name

        try:
            os.mkdir(path)
        except OSError as error:
            print(error)

        plt.figure(figsize=(16,8))
        plt.title("Training Loss")
        results = self.model.evals_result()
        plt.plot(results['validation_0']['mae'], label='train')
        plt.plot(results['validation_1']['mae'], label='test')
        plt.legend()
        plt.savefig(path + '/loss_plot.png')
        plt.close()

        plt.figure(figsize=(20, 8))
        plt.title("Feature Importances")
        plt.bar(range(X_train.shape[1]),
                importances[indices], color="r", align="center")
        plt.xticks(range(X_train.shape[1]),
                   X_train.columns[indices], rotation=90)
        plt.xlim([-1, X_train.shape[1]])
        plt.savefig(path + '/features_importances.png')
        plt.tight_layout()
        plt.close()

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test_t)
        shap.summary_plot(shap_values, X_test_t, feature_names=X_test.columns,
                          max_display=50, plot_size=(16, 20), show=False)
        plt.savefig(path + '/feature_importances_shap.png')
        plt.close()

        pd.DataFrame([self.model.get_params()]).to_csv(f"{path}/best_params.csv")

    def save_model(self, pathname):
        pkl_filename = pathname + '.pkl'

        with open(pkl_filename, 'wb') as file:
            pickle.dump(self.model, file)
