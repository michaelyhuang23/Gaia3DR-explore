import numpy as np
import pandas as pd
import os

root = 'caterpillar/labeled_caterpillar_data'
for file in os.listdir(root):
    if '.h5' not in file : continue
    file = os.path.join(root, file)
    df = pd.read_hdf(file, key='star')
    df = df.loc[~df['jzstar'].isnull()]
    df.to_hdf(file, key='star')
