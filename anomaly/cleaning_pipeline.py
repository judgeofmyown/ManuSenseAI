import yaml
import pandas as pd
from cleaner import (
    standardize, drop_cols, reset_index, drop_low_std_cols, fill_missing
)

def run_cleaning_pipeline(df_key: str, df: pd.DataFrame, config: dict) -> pd.DataFrame:
    
    """Run the cleaning pipeline based on the processing pipeline"""

    if(config.get("drop_low_std_cols", {}).get("enabled", False)):
        threshold = config["drop_low_std_cols"].get("std_threshold", 1e-2)
        df = drop_low_std_cols(df, threshold)
    
    if(config.get("fill_missing", {}).get("enabled", False)):
        method = config["fill_missing"].get("method", "ffill")
        df = fill_missing(df, method)
    
    if config.get("standardize", {}).get("enabled", False):
        df = standardize(df)
    
    if config.get("drop_cols", {}).get("enabled", False):
        features_to_delete = config["drop_cols"].get("features_to_delete", {})
        cols = features_to_delete.get(df_key, []) if df_key else []
        df = drop_cols(df, cols)
    
    if config.get("reset_index", {}).get("enabled", False):
        df = reset_index(df)
    
    return df