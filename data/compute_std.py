import pandas as pd
import numpy as np
import os


data_dir = 'simulation'

file_paths = [os.path.join(data_dir,file) for file in os.listdir(data_dir) if file.endswith('.h5')]
for path in file_paths:
    if 'tree' in path: continue
    print(f'reading {path}')
    df = pd.read_hdf(path, key='star')
    df_std = pd.DataFrame()
    for column in df.columns:
        df_std[column] = np.array([np.std(df[column].to_numpy())])
    df_std.to_csv(path.split('.')[0]+'_std.csv')

