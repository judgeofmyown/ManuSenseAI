import pandas as pd
import logging
import yaml
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path: str = "configs/default_config.yaml") -> dict:
    "Load YAML config from a file"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_column_names(config: dict) -> list | None:
    """load columns names from a YAML files"""
    col_file = config['data'].get('column_names_path')
    if col_file and os.path.exists(col_file):
        with open(col_file) as f:
            names_dict = yaml.safe_load(f) or {} # fallback to {}
        
        col_names = names_dict.get('columns_names')
        if not col_names or not isinstance(col_names, list):
            logging.warning(f"'columns_names' is empty or invalid in {col_file}")
            return None
        return col_names
    return None

def save_column_names(columns: list, path: str):
    """Save a list of column names to a YAML file"""
    with open(path, 'w') as f:
        yaml.dump({'columns_names': columns}, f)
    logging.info(f"Saved inferred column names to: {path}")

def load_data(config: dict) -> pd.DataFrame:
    """Main data loader function based on YAML config"""
    data_config = config['data']
    path = data_config['input_path']
    has_header = data_config.get('has_header', True)
    col_path = data_config.get('column_names_path')
    save_cols = data_config.get('save_inferred_columns', False)

    col_names = get_column_names(config) if not has_header else None

    if not os.path.exists(path):
        logging.error(f"File not found: {path}")
        raise FileNotFoundError(path)

    df = pd.read_csv(path, header=None if not has_header else 'infer')

    # handle header-less case
    if not has_header:
        if col_names:
            if len(col_names) != df.shape[1]:
                raise ValueError(f"Expected {len(col_names)} columns, got {df.shape[1]}")
            df.columns = col_names
        else:
            # inferr col names
            inferred_cols = [f"col_{i}" for i in range(df.shape[1])]
            df.columns = inferred_cols
            if save_cols and col_path:
                save_column_names(inferred_cols, col_path) 
    
    logging.info(f"Data loaded successfully. Shape: {df.shape}")
    
    return df
