import numpy as np
import pandas as pd
import os

data_dir = 'caterpillar/labeled_caterpillar_data'

file_paths = [os.path.join(data_dir,file) for file in os.listdir(data_dir) if file.endswith('.h5')]
for path in file_paths:
    if 'tree' in path: continue
    print(f'reading {path}')
    df = pd.read_hdf(path, key='star')
    df['rstar'] = np.linalg.norm([df['xstar'], df['ystar'], df['zstar']], axis=0)
    
    df.to_hdf(path, key='star')
    
