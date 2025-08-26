import pandas as pd
import logging
import yaml
import re
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path: str = "configs/default_config.yaml") -> dict:
    """Load YAML config from a file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_column_names(config: dict) -> list | None:
    """load columns names from a YAML files"""
    return config.get("metadata", {}).get("features", None)

def save_column_names(columns: list, path: str):
    """Save a list of column names to a YAML file"""
    with open(path, 'w') as f:
        yaml.dump({'columns_names': columns}, f)
    logging.info(f"Saved inferred column names to: {path}")

def load_data_from_config(config: dict, key: str) -> pd.DataFrame:
    """
     Loads multiple CMAPSS files based on YAML config.
    Args:
        config: full YAML configuration dict
        split: 'train' or 'test'
    Returns:
        Concatenated DataFrame of all files
    """

    match = re.match(r"(train|test)_FD\d{3}", key)
    if not match:
        raise ValueError(f"Invalid key format: {key}. Expected format like 'train_FD001'.")

    split, fd = key.split("_")
    fd_number = fd.upper()  # 'FD001'

    # Find the correct path
    paths = config["data_path"].get(split, [])
    selected_path = None
    for path in paths:
        if fd_number in path:
            selected_path = path
            break

    if not selected_path:
        raise FileNotFoundError(f"No file found in config['data_path']['{split}'] matching {fd_number}")

    # Load columns from metadata
    col_names = config["metadata"]["features"]
    if not col_names:
        raise ValueError("Missing column names in config['metadata']['features']")

    if not os.path.exists(selected_path):
        raise FileNotFoundError(f"Path does not exist: {selected_path}")

    df = pd.read_csv(selected_path, sep=r"\s+", header=None)
    df = df.loc[:, ~df.columns.duplicated()]
    if len(df.columns) != len(col_names):
        raise ValueError(f"Expected {len(col_names)} columns, got {len(df.columns)} in {selected_path}")

    df.columns = col_names
    df["source"] = key  # Optional: Add source label like 'train_FD001'

    logging.info(f"{key} loaded. Shape: {df.shape}")
    return df

class CMAPSSDataLoader:
    """Base Data loader class for CMAPSS datsets"""

    def __init__(self, df: pd.DataFrame, config: dict = None):
        self.df = df
        self.machine_ids = self.df["number"].unique()
        self._groups = self.df.groupby("number")
        self.signal_columns = [col for col in self.df.columns if col.startswith("sensor-")]
    
    def get_machine(self, machine_id: int) -> pd.DataFrame:
        """Get data for a specific machine"""
        if machine_id not in self.machine_ids:
            raise ValueError(f"Machine ID {machine_id} not found in dataset")
        return self._groups.get_group(machine_id)
    
    def __iter__(self):
        """Iterates over (machine_id, machine_df) pairs"""
        for machine_id, machine_df in self._groups:
            yield machine_id, machine_df
    
    def __len__(self):
        """Number of unique machines"""
        return len(self.machine_ids)
    
    def get_all(self):
        """Return full dataframe"""
        return self.df.copy()
    
    def get_metadata(self):
        raise NotImplementedError("Metadata extraction not implemented yet")

class CMAPSSWindowDataLoader(CMAPSSDataLoader):
    """
    Data loader for CMAPSS datasets for windowed inputs
    ---------
    Assumes the dataset has required features only
    """

    def __init__(self, df: pd.DataFrame, window_size: int = 10, stride: int = 1):
        super().__init__(df)
        self.window_size = window_size
        self.stride = stride
    
    def iter_windows(self):
        """Iterate over windows for each machine"""

        for machine_id, machine_df in self.iter_windows():
            n_rows = len(machine_df)
            for start in range(0, n_rows - self.window_size + 1, self.stride):
                end = start + self.window_size
                window_df = machine_df.iloc[start:end][self.signal_columns].values
                yield machine_id, start, window_df
    
    def get_machine_windows(self, machine_id: int):
        """Get all windows for a specific machine"""
        if machine_id not in self.machine_ids:
            raise ValueError(f"Machine ID {machine_id} not found in dataset")

        machine_df = self.get_machine(machine_id).reset_index(drop=True)
        n_rows = len(machine_df)
        windows = []
        for start in range(0, n_rows - self.window_size + 1, self.stride):
            window = machine_df.iloc[start: start  + self.window_size]
            windows.append(window)
        return windows

