import numpy as np
import matplotlib.pyplot as plt


def plot_series(time, series, kind="-", start=0, end=None, label=None):
    """
    Standard plotting function with matplotlib for visualising series.
    """
    plt.plot(time[start:end], series[start:end], kind, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.xticks(fontsize=12, alpha=.7)
    plt.yticks(fontsize=12, alpha=.7)
    if label:
        plt.legend(fontsize=14)
    plt.grid(axis='both', alpha=.3)


def trend(time, slope=0):
    """
    Function to add an increasing or decreasing trend to series.
    use slope of a line to control the trend.
    line is defined as y = slope * x + intercept. Where, intercept = 0.
    """
    return slope * time


def seasonal_pattern(season_time):
    """
    Generates an arbitrary pattern for seasonality.
    Series generates a cosine trend followed by exponential drop.
    """
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def white_noise(time, noise_level=1, seed=None):
    """
    Add random noise to series.
    """
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def autocorrelation(time, amplitude, seed=None):
    """
    Add some autocorrelation to the series.
    """
    rnd = np.random.RandomState(seed)
    φ1 = 0.5
    φ2 = -0.1
    ar = rnd.randn(len(time) + 50)
    ar[:50] = 100
    for step in range(50, len(time) + 50):
        ar[step] += φ1 * ar[step - 50]
        ar[step] += φ2 * ar[step - 33]
    return ar[50:] * amplitude
