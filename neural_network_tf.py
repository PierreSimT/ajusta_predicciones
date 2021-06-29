import os

import logging
import tensorflow as tf

from shap import DeepExplainer, summary_plot

from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import Hyperband

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

from Model import Model

class NeuralNetworkTF(Model):

    name = "NeuralNetwork"

    def __init__(self, data, n_outputs):

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
        self.n_neurons = data.shape[1] - n_outputs     
        self.n_outputs = n_outputs   

        inputs = keras.Input(shape=(self.n_neurons,))
        dense1 = layers.Dense(self.n_neurons, activation='tanh')
        dense2 = layers.Dense(self.n_neurons, activation='tanh')
        dense3 = layers.Dense(self.n_neurons, activation='tanh')
        dense4 = layers.Dense(self.n_neurons, activation='tanh')
        dense5 = layers.Dense(self.n_outputs)

        x = dense1(inputs)
        x = dense2(x)
        x = dense3(x)
        x = dense4(x)
        output = dense5(x)

        self.model = keras.Model(inputs=inputs, outputs=output, name="WRFModel")
        self.model.compile(loss=tf.losses.MeanAbsoluteError(), optimizer=tf.optimizers.Adam())
        self.model.summary()

        
    def train(self, X_train, X_test, y_train, y_test, output_directory, n_epochs):
        """Entrenamiento del modelo

        Args:
            X_train (DataFrame): Conjunto de entrenamiento
            X_test (DataFrame): Conjunto de validacion
            y_train (DataFrame): Resultados del conjunto de entrenamiento
            y_test (DataFrame): Resultados del conjunto de validacion
            n_epochs (int): Numero de epochs en caso de entrenar una red neuronal
        """

        path = output_directory + '/' + self.name

        try:
            os.mkdir(path)
        except OSError as error:
            print(error)

        X_train_t = self.scaler.fit_transform(X_train)
        X_test_t = self.scaler.transform(X_test)

        logging.info("Neural Network TF: Training Model")
        history = self.model.fit(X_train_t, y_train, validation_data=(X_test_t, y_test), epochs=n_epochs)

        print("Saving History")
        loss_values = history.history['loss']
        val_loss_values = history.history['val_loss']
        epochs = range(1, len(loss_values) + 1)

        plt.figure(figsize=(18,8))
        plt.title("Training Loss")
        plt.plot(loss_values, label="train")
        plt.plot(val_loss_values, label="test")
        plt.legend()
        plt.savefig(path + '/loss_plot.png')
        plt.close()

        print("Saving importances")
        explainer = DeepExplainer(self.model, X_test_t)
        shap_values = explainer.shap_values(X_test_t)

        summary_plot(shap_values, X_test_t, feature_names=X_test.columns, max_display=50, plot_size=(16, 20), show=False)
        plt.savefig(path + '/feature_importances.png')
        plt.close()

        
    
    def save_model(self, pathname):
        return self.model.save(pathname + '.h5')

    def get_model(self):
        return self.model
    

