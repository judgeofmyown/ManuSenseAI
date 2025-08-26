import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

"""
# composing pipelines

processing_steps = [
    (compute_rms, "rms10", {"window": 10}),
    (smooth_lowess, "lowess05", {"frac": 0.05}),
]

for func, suffix, params in processing_steps:
    for col in target_cols:
        df = apply_and_name(df, col, func, suffix, **params)
"""

# cleaning data
def drop_low_std_cols(df: pd.DataFrame, std_threshold=1e2) -> pd.DataFrame:
    stds = df.std()
    to_drop = stds[stds < std_threshold].index
    return df.drop(columns=to_drop)

def fill_missing(df: pd.DataFrame, method='ffill') -> pd.DataFrame:
    return df.fillna(method=method)

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df_scaled

def drop_cols(df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    return df.drop(columns=cols_to_drop)

def reset_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index(drop=True)