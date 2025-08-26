import pandas as pd

# unified interface
def apply_and_name(df: pd.DataFrame, col: str, func, suffix: str, **kwargs):
    new_col = f"{col}_{suffix}"
    df[new_col] = func(df[col], **kwargs)
    return df