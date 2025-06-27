from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
import numpy as np
import os

# Time domain techniques
def rolling_z_score(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return (series - mean)/std

# ----technique 1 [rolling rms + lowess]
def rolling_rms(series: pd.Series, window: int) -> pd.Series:
    return (series**2).rolling(window=window).mean().apply(np.sqrt)

def smooth_lowess(series: pd.Series, frac=0.05) -> pd.Series:
    x = np.arange(len(series))
    y = series.values
    smoothed = lowess(y, x, frac=frac, return_sorted=False)
    return pd.Series(smoothed, index=series.index)