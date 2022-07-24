import pandas as pd
import numpy as np
import json
import os

data = {}
for file in os.listdir('.'):
    if not file.endswith('.json'): continue
    with open(file, 'r') as f:
        result = json.load(f)
        result_arr = [result['IoU_recall'], result['IoU_precision'], result['Mode_recall'], result['Mode_precision'], result['ARand'], result['AMI'], result['Purity'], result['Mode_F1'], result['Mode_recall_C']]
        data[int(file.split('.')[0].split('_')[-1])] = result_arr

df = pd.DataFrame(data)
df = df.reindex(sorted(df.columns), axis=1)
print(df.head())

df.to_csv('collected_data.csv')
        
