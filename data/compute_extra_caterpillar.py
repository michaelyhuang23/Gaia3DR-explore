import numpy as np
import pandas as pd
import os

data_dir = 'caterpillar/labeled_caterpillar_data'

file_paths = [os.path.join(data_dir,file) for file in os.listdir(data_dir) if file.endswith('.h5')]
for path in file_paths:
    if 'tree' in path: continue
    print(f'reading {path}')
    df = pd.read_hdf(path, key='star')
    df['rstar'] = np.linalg.norm([df['xstar'], df['ystar']], axis=0)
    df['phistar'] = np.arctan2(df['ystar'], df['xstar'])
    rvec = np.stack([df['xstar'], df['ystar']], axis=-1)
    irvec = rvec / np.linalg.norm(rvec, axis=-1)[...,None]
    vvec = np.stack([df['vxstar'], df['vystar']], axis=-1)
    df['vrstar'] = np.sum(irvec * vvec, axis=-1)
    df['vphistar'] = np.cross(irvec, vvec)
    df['vstar'] = np.linalg.norm([df['vxstar'], df['vystar'], df['vzstar']], axis=0)
    df.to_hdf(path, key='star')
    
