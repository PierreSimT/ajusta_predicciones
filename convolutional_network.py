import os

import logging
import tensorflow as tf
import shap

from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import Hyperband


from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

from Model import Model

class ConvolutionalNetwork(Model):

    name = "ConvolutionalNetwork"

    def __init__(self, data, n_outputs):

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
        self.n_neurons = data.shape[1] - n_outputs
        self.n_outputs = n_outputs

        # ( numero_de_datos, features ) -> (num_de_datos, 1, features, 1 )
        print(data)

        inputs = keras.Input(shape=(self.n_neurons,))
        reshape = layers.Reshape((1, self.n_neurons, 1))
        conv1 = layers.Conv2D(12, (1, 2), padding="same", activation="tanh")
        # reshape2 = layers.Reshape((n_neurons, 6, 2))
        poolin1 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")
        conv2 = layers.Conv2D(6, (1, 2), padding="same", activation="tanh") 
        flat = layers.Flatten()
        dense1 = layers.Dense(12 * self.n_neurons, activation='tanh')
        dense2 = layers.Dense(self.n_neurons)
        dense3 = layers.Dense(self.n_outputs)

        x = reshape(inputs)
        x = conv1(x)
        # x = poolin1(x)
        x = conv2(x)
        x = flat(x)
        x = dense1(x)
        x = dense2(x)
        output = dense3(x)

        self.model = keras.Model(inputs=inputs, outputs=output, name="WRFModelConv")
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

        logging.info("Convolutional Network TF: Training Model")
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
        explainer = shap.KernelExplainer(self.model, X_test_t)
        shap_values = explainer.shap_values(X_test_t)

        shap.summary_plot(shap_values, X_test_t, feature_names=X_test.columns, max_display=50, plot_size=(16, 20), show=False)
        plt.savefig(path + '/feature_importances.png')
        plt.close()
        
    
    def save_model(self, pathname):
        return self.model.save(pathname + '.h5')

    def get_model(self):
        return self.model
    

