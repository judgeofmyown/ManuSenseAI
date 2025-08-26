import pandas as pd
import numpy as np

# ----technique 2 [auto correlation features]
def autocorr_manual(series: pd.Series, lag: int) -> float:
    """
    Compute the autocorrelation of a time series at a specified lag using the manual formula.

    Autocorrelation at lag `k` measures the linear relationship between the time series and its 
    lagged version by `k` steps. This function assumes the series is stationary, meaning its 
    mean and variance are constant over time.

    Parameters:
    ----------
    series : pd.Series
        The time series data to compute autocorrelation on.
    lag : int
        The number of time steps to lag the series.

    Returns:
    -------
    float
        The autocorrelation coefficient at the specified lag.
        Returns NaN if the lag is greater than or equal to the length of the series or if variance is zero.
    """
    if lag > len(series):
        return np.nan
    x = series.values
    mean = np.mean(x)
    numerator = np.sum((x[:-lag] - mean)*(x[lag:] - mean))
    denominator =  np.sum((x - mean)**2)
    return numerator/denominator if denominator != 0 else np.nan

def autocorr_lib(series: pd.Series, lag: int) -> float:
    return series.autocorr(lag=lag)

def rolling_autocorr(series: pd.Series, lag: int, window: int) -> pd.Series:
    """applies rolling autocorr accross a dataframe's column"""
    return series.rolling(window=window).apply(lambda x: autocorr_manual(x, lag), raw=False)