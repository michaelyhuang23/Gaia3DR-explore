import pandas as pd
import numpy as np
import json
import os


data_dir = 'simulation'

file_paths = [os.path.join(data_dir,file) for file in os.listdir(data_dir) if file.endswith('.h5')]
for path in file_paths:
    if 'tree' in path: continue
    print(f'reading {path}')
    df = pd.read_hdf(path, key='star')
    df_norm = {'mean':{}, 'std':{}}
    for column in df.columns:
        df_norm['std'][column] = float(np.std(df[column]))
        df_norm['mean'][column] = float(np.mean(df[column]))
    with open(path.split('.')[0]+'_norm.json', 'w') as f:
        json.dump(df_norm, f)

