import numpy as np
import pandas as pd
import os

data_dir = 'simulation'

file_paths = [os.path.join(data_dir,file) for file in os.listdir(data_dir) if file.endswith('.h5')]
for path in file_paths:
    if 'tree' in path: continue
    print(f'reading {path}')
    df = pd.read_hdf(path, key='star')
    df['rstar'] = np.linalg.norm([df['xstar'], df['ystar'], df['zstar']], axis=0)
    df['s_jzrstar'] = df['jzstar'].to_numpy() - df['jrstar'].to_numpy()
    df['a_jzrstar'] = df['jzstar'].to_numpy() + df['jrstar'].to_numpy()
    df['c_lzstar'] = df['lzstar'].to_numpy() * np.abs(df['estar'].to_numpy())**2
    df['c_lxstar'] = df['lxstar'].to_numpy() * np.abs(df['estar'].to_numpy())**2
    df['c_lystar'] = df['lystar'].to_numpy() * np.abs(df['estar'].to_numpy())**2
    df['c_jzstar'] = df['jzstar'].to_numpy() * np.abs(df['estar'].to_numpy())**2
    df['c_jrstar'] = df['jrstar'].to_numpy() * np.abs(df['estar'].to_numpy())**2
    df.to_hdf(path, key='star')
    
