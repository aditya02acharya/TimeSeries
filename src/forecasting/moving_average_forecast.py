import numpy as np


class Moving_Average():
    """
    Forecasting class that does time series prediction by generating a series
    that does a windowed averaging. Additional bits and bobs includes averaging
    by removing seasonality and trend.
    """

    def __init__(self, series, window_size, augmentation=False, period=None):
        self.series = series
        self.window_size = window_size
        self.augmentation = augmentation
        self.augmented_series = None
        self.period = period

        if self.augmentation:
            self.augmented_series = (self.series[self.period:] - self.series[:-self.period])

    def moving_average_forecast(self):
        """Forecasts the mean of the last few values.
           If window_size=1, then this is equivalent to naive forecast"""
        forecast = []

        if self.augmentation:
            for time in range(len(self.augmented_series) - self.window_size):
                forecast.append(self.augmented_series[time:time + self.window_size].mean())
        else:
            for time in range(len(self.series) - self.window_size):
                forecast.append(self.series[time:time + self.window_size].mean())

        return np.array(forecast)
