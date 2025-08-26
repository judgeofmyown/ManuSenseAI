TRAIN_PATH = [  
    "Data\CMAPSSData\train_FD001.txt",
    "Data\CMAPSSData\train_FD002.txt",
    "Data\CMAPSSData\train_FD003.txt", 
    "Data\CMAPSSData\train_FD004.txt"
]

DROP_COLS_001 = [4, 3, 22, 23, 20, 19, 14, 9, 5, 10] # based on EDA (only done on train_fd001) these columns must be dropped

DATA_COLUMS_001 = [      
    'number', 'time', 'ops-set-1', 'sensor_6',
    'sensor_7', 'sensor_8', 'sensor_11', 'sensor_12', 
    'sensor_13', 'sensor_15', 'sensor_16', 'sensor_17', 
    'sensor_18', 'sensor_21', 'sensor_24', 'sensor_25'
]

