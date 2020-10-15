import argparse
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from src.forecasting.rnn_forecasting import RNN_Forecast
from src.forecasting.naive_forecast import Naive_Forecast
from src.forecasting.moving_average_forecast import Moving_Average
from src.synthetic_data.series import trend, plot_series, seasonality, white_noise

parser = argparse.ArgumentParser(description="Time Series Playground")

series_args = parser.add_argument_group("Series Type")
series_args.add_argument("--trend", action="store_true", default=False,
                         help="generate synthetic series with a trend.")
series_args.add_argument("--seasonal", action="store_true", default=False,
                         help="generate synthetic series with a seasonality.")
series_args.add_argument("--noise", action="store_true", default=False,
                         help="generate synthetic series with a noise.")

data_args = parser.add_argument_group("Series Meta Data")
data_args.add_argument("--slope", type=float, default=0.05, help="slope value for the trend.")
data_args.add_argument("--amplitude", type=float, default=40.0, help="slope value for the trend.")
data_args.add_argument("--period", type=int, default=365, help="noise magnitude.")
data_args.add_argument("--noise_level", type=float, default=5.0, help="noise magnitude.")
data_args.add_argument("--seed", type=int, default=51, help="random seed value.")

forecast_args = parser.add_argument_group("Forecasting")
forecast_args.add_argument("--naive_forecast", action="store_true", default=False,
                           help=" generates series using naive forecast that lags 1 step behind the time series.")
forecast_args.add_argument("--moving_avg", action="store_true", default=False,
                           help=" generates series using n-step moving average.")
forecast_args.add_argument("--rnn_forecast", action="store_true", default=False,
                           help=" generates series using recurrent network.")

data_args = parser.add_argument_group("Forecasting Meta Data")
data_args.add_argument("--lag_size", type=int, default=1, help="lag size for naive forecasting.")
data_args.add_argument("--window_size", type=int, default=20, help="window size for moving average forecasting.")
data_args.add_argument("--augment", action="store_true", default=False, help="remove seasonality and trend.")
data_args.add_argument("--batch_size", type=int, default=32, help="window size for moving average forecasting.")

args = parser.parse_args()
tf.random.set_seed(args.seed)
np.random.seed(args.seed)


def get_time():
    return np.arange(4 * args.period + 1)


def split_train_test(time_split=1000):
    t_train = time[:time_split]
    train = series[:time_split]
    t_valid = time[time_split:]
    valid = series[time_split:]

    return t_train, train, t_valid, valid


def windowed_dataset(data, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


if args.trend and not args.seasonal and not args.noise:
    time = get_time()
    series = trend(time, args.slope)

    plt.figure(figsize=(10, 6))
    plot_series(time, series)
    plt.show()

if args.seasonal and not args.trend and not args.noise:
    time = get_time()
    series = seasonality(time, period=args.period, amplitude=args.amplitude)

    plt.figure(figsize=(10, 6))
    plot_series(time, series)
    plt.show()

if args.noise and not args.trend and not args.seasonal:
    time = get_time()
    noise = white_noise(time, args.noise_level, seed=args.seed)

    plt.figure(figsize=(10, 6))
    plot_series(time, noise)
    plt.show()

if args.seasonal and args.trend and not args.noise:
    time = get_time()
    series = trend(time, args.slope) + seasonality(time, period=args.period, amplitude=args.amplitude)

    plt.figure(figsize=(10, 6))
    plot_series(time, series)
    plt.show()

if args.seasonal and args.noise and not args.trend:
    time = get_time()
    series = seasonality(time, period=args.period, amplitude=args.amplitude) + \
             white_noise(time, args.noise_level, seed=args.seed)

    plt.figure(figsize=(10, 6))
    plot_series(time, series)
    plt.show()

if args.noise and args.trend and not args.seasonal:
    time = get_time()
    series = trend(time, args.slope) + white_noise(time, args.noise_level, seed=args.seed)

    plt.figure(figsize=(10, 6))
    plot_series(time, series)
    plt.show()

if args.seasonal and args.trend and args.noise:
    time = get_time()
    series = trend(time, args.slope) + seasonality(time, period=args.period, amplitude=args.amplitude) + \
             white_noise(time, args.noise_level, seed=args.seed)

    plt.figure(figsize=(10, 6))
    plot_series(time, series)
    plt.show()

if args.naive_forecast:
    time = get_time()
    series = trend(time, args.slope) + seasonality(time, period=args.period, amplitude=args.amplitude) + \
             white_noise(time, args.noise_level, seed=args.seed)

    split_time = 1000
    time_train, x_train, time_valid, x_valid = split_train_test(split_time)

    predictor = Naive_Forecast(series=series)

    naive_forecast_series = predictor.predict_series(split_time)

    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid, label="actual")
    plot_series(time_valid, naive_forecast_series, label="predicted")
    plt.show()

if args.moving_avg:
    time = get_time()
    series = trend(time, args.slope) + seasonality(time, period=args.period, amplitude=args.amplitude) + \
             white_noise(time, args.noise_level, seed=args.seed)

    split_time = 1000
    time_train, x_train, time_valid, x_valid = split_train_test(split_time)

    predictor = Moving_Average(series, args.window_size, args.augment, args.period)

    if args.augment:
        diff_moving_avg = predictor.moving_average_forecast()[split_time - args.period - args.window_size:]
        moving_avg = series[split_time - args.period:-args.period] + diff_moving_avg
    else:
        moving_avg = predictor.moving_average_forecast()[split_time - args.window_size:]

    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid, label="actual")
    plot_series(time_valid, moving_avg, label="predicted")
    plt.show()

if args.rnn_forecast:
    time = get_time()
    series = trend(time, args.slope) + seasonality(time, period=args.period, amplitude=args.amplitude) + \
             white_noise(time, args.noise_level, seed=args.seed)

    split_time = 1000
    time_train, x_train, time_valid, x_valid = split_train_test(split_time)

    tf_data = windowed_dataset(x_train, args.window_size, args.batch_size, 1000)

    predictor = RNN_Forecast(tf_data)

    history = predictor.train()

    forecast = []
    results = []
    for time in range(len(series) - args.window_size):
        forecast.append(predictor.model.predict(series[time:time + args.window_size][np.newaxis]))

    forecast = forecast[split_time - args.window_size:]
    results = np.array(forecast)[:, 0, 0]

    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid, label="actual")
    plot_series(time_valid, results, label="predicted")
    plt.show()
