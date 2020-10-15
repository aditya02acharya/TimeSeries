class Naive_Forecast():
    """
    Forecasting class that does time series prediction by generating a series
    that lags n step behind the original time series.
    """

    def __init__(self, n_step_lag=1, series=None):
        self.n_step_lag = n_step_lag
        self.series = series

    def predict_series(self, timestep):
        forecast = self.series[timestep - self.n_step_lag: -self.n_step_lag]
        return forecast
