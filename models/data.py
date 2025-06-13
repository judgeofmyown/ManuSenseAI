import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess

train_path = "./notebooks/CMAPSSData/train_FD001.txt"
standard_scaler = StandardScaler()
train_data = pd.read_csv(train_path, delim_whitespace=True, header=None)
cols_to_dop = [4, 3, 22, 23, 20, 19, 14, 9, 5, 10]
train_data_column = ['number', 'time', 'ops-set-1', 'sensor_6',
                     'sensor_7', 'sensor_8', 'sensor_11', 'sensor_12', 
                     'sensor_13', 'sensor_15', 'sensor_16', 'sensor_17', 
                     'sensor_18', 'sensor_21', 'sensor_24', 'sensor_25']



def process_data(data):
    data = data.drop(cols_to_dop, axis=1)
    data.columns = train_data_column
    sensor_cols = data.columns[3:]

    for sensor in sensor_cols:
        # rolling rms calc.
        data[f"{sensor}_rolling_rms"] = data[sensor].rolling(window=10).apply(lambda x: np.sqrt(np.mean(x**2)))

    data = data.fillna(method='bfill') # repalcing NaNs caused due to rolling

    original_cols = train_data_column[3:]
    data = data.drop(original_cols, axis=1)

    # scaling
    features_to_scale = data.columns[2:]
    data[features_to_scale] = standard_scaler.fit_transform(data[features_to_scale])

    # RUL Labeling
    max_cycle_per_unit = data.groupby('number')['time'].max()
    data['max_cycle'] = data['number'].map(max_cycle_per_unit)

    data['RUL'] = data['max_cycle'] - data['time']
    data = data.drop(columns=['max_cycle'], axis=1)

    return data

def apply_lowess_slope(group, sensor_cols):
    for col in sensor_cols:
        # apply lowess smoothing 
        smoothed = lowess(endog=group[col], exog=group['time'], frac=0.5, return_sorted=False)

        slope = np.gradient(smoothed)

        group[f"{col}_lowess_slope"] = slope
    return group

def add_rolling_slope(data):
    sensor_cols = [col for col in data.columns if '_rms' in col]

    data = data.groupby('number', group_keys=False).apply(lambda g: apply_lowess_slope(g, sensor_cols))

    slope_cols = [col for col in data.columns if 'lowess_slope' in col]
    data[slope_cols] = standard_scaler.fit_transform(data[slope_cols])
    # test
    return data

print("processing data ...")
data = process_data(train_data)
print('adding rolling slope ...')
data = add_rolling_slope(data)
print('saving ...')
data.to_csv('data.csv', index=False)

