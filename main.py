import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.synthetic_data.series import trend, plot_series, seasonality, white_noise

parser = argparse.ArgumentParser(description="Time Series Playground")

series_args = parser.add_argument_group("Series Type")
series_args.add_argument("--trend", action="store_true", default=False,
                         help="generate synthetic series with a trend.")
series_args.add_argument("--seasonal", action="store_true", default=False,
                         help="generate synthetic series with a seasonality.")
series_args.add_argument("--noise", action="store_true", default=False,
                         help="generate synthetic series with a noise.")
series_args.add_argument("--autocorr", action="store_true", default=False,
                         help="generate synthetic series with a autocorrelation.")

data_args = parser.add_argument_group("Series Meta Data")
data_args.add_argument("--slope", type=float, default=0.1, help="slope value for the trend.")
data_args.add_argument("--amplitude", type=float, default=40.0, help="slope value for the trend.")
data_args.add_argument("--noise_level", type=float, default=1.0, help="noise magnitude.")
data_args.add_argument("--seed", type=int, default=10, help="random seed value.")

args = parser.parse_args()

if args.trend and not args.seasonal and not args.noise and not args.autocorr:
    time = np.arange(4 * 365 + 1)
    series = trend(time, args.slope)

    plt.figure(figsize=(10, 6))
    plot_series(time, series)
    plt.show()

if args.seasonal and not args.trend and not args.noise and not args.autocorr:
    time = np.arange(4 * 365 + 1)
    series = seasonality(time, period=365, amplitude=args.amplitude)

    plt.figure(figsize=(10, 6))
    plot_series(time, series)
    plt.show()

if args.noise and not args.trend and not args.seasonal and not args.autocorr:
    time = np.arange(4 * 365 + 1)
    noise = white_noise(time, args.noise_level, seed=args.seed)

    plt.figure(figsize=(10, 6))
    plot_series(time, noise)
    plt.show()

if args.seasonal and args.trend and not args.noise and not args.autocorr:
    time = np.arange(4 * 365 + 1)
    series = trend(time, args.slope) + seasonality(time, period=365, amplitude=args.amplitude)

    plt.figure(figsize=(10, 6))
    plot_series(time, series)
    plt.show()

if args.seasonal and args.trend and args.noise:
    time = np.arange(4 * 365 + 1)
    series = trend(time, args.slope) + seasonality(time, period=365, amplitude=args.amplitude) \
             + white_noise(time, args.noise_level, seed=args.seed)

    plt.figure(figsize=(10, 6))
    plot_series(time, series)
    plt.show()


