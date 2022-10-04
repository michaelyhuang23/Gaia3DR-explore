import pandas as pd
import numpy as np
import os

data = {}
root = 'caterpillar/caterpillar_data'
for file in os.listdir(root):
    name, num = file.split('.')[0].split('_')
    file_path = os.path.join(root, file)
    df = pd.read_hdf(file_path, key='star')
    
    df.sort_values('idstar', inplace=True, kind='stable')
    prev_val = -1
    prev_id = 0
    cluster_ids = np.zeros((len(df)), dtype=np.int32)
    for i, (idx, row) in enumerate(df.iterrows()):
        if (row['idstar'] - prev_val)>0.1:
            prev_id+=1
            prev_val = row['idstar']
        cluster_ids[i] = prev_id
    df['cluster_id'] = cluster_ids
    print(prev_id)
    if name not in data.keys():
        data[name] = {}
    data[name][num] = df

for key in data.keys():
    max_id = 0
    data[key]['all'] = pd.DataFrame(columns = data[key]['0'].columns)
    data[key]['all']['cluster_id'] = data[key]['all']['cluster_id'].astype('int32')
    for i in range(0,5):
        df = data[key][str(i)].copy()
        df['cluster_id'] += max_id
        if len(df)>0:
            data[key]['all'] = pd.concat([data[key]['all'], df])
        max_id = data[key]['all']['cluster_id'].max()
    print(data[key]['0']['cluster_id'].max(), data[key]['1']['cluster_id'].max(), data[key]['2']['cluster_id'].max(),\
          data[key]['3']['cluster_id'].max(), data[key]['4']['cluster_id'].max(), data[key]['all']['cluster_id'].max())


output = 'caterpillar/labeled_caterpillar_data'
for key in data.keys():
    for key2 in data[key].keys():
        data[key][key2].to_hdf(os.path.join(output, f'labeled_{key}_{key2}.h5'), key='star')
