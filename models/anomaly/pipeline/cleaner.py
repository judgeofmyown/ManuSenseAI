import pandas as pd
import numpy as np
import os

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