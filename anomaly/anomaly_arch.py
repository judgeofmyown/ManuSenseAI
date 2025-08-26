import pandas as pd
import numpy as np
from data_loader import (
    load_data_from_config, load_config, CMAPSSWindowDataLoader
)

dataset_config = load_config("config/dataset.yaml")
processing_config = load_config("config/processing.yaml")

df_train = load_data_from_config(dataset_config, "train")
df_test = load_data_from_config(dataset_config, "test")

train_window_loader = CMAPSSWindowDataLoader(df_train, window_size=30, stride=5)
test_window_loader = CMAPSSWindowDataLoader(df_test, window_size=30, stride=5)

# do anomaly detection on per machine
# each machine id:
#           dataframe of machine sensors [1, 2, 3, ...]
#           each windowed data of each sensor : S1[i:i+w], S2[i:i+w], S3[i:i+w], .... , SN[i:i+w]
#           for each Sj[i:i+w]
#               input to FseNET model for feature/latent space
#               use this latent space for anomaly detection


