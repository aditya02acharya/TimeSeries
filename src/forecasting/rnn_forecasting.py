import numpy as np
import tensorflow as tf


class RNN_Forecast():
    """
    Forecasting class that does time series prediction by using DNN.
    Specifically, model uses stacked LSTM network.
    """

    def __init__(self, dataset=None):
        self.dataset = dataset

        # model description.
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                                   input_shape=[None]),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Lambda(lambda x: x * 100.0)
        ])

        self.model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9), metrics=["mae"])

    def predict_series(self, timestep):
        forecast = self.series[timestep - self.n_step_lag: -self.n_step_lag]
        return forecast

    def train(self):
        history = self.model.fit(self.dataset, epochs=500, verbose=1)
        return history
