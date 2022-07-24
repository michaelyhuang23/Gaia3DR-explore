import torch
import seaborn as sns
import os
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from collections import Counter
from pyvis.network import Network
from random import randint
import sklearn
import json

from neural_dataset import ClusterDataset
from cluster_analysis import C_HDBSCAN,C_Spectral,C_GaussianMixture
from evaluation_metrics import ClusterEvalAll
from utils import cart2spherical, UnionFind

# apply on m12f, scale of m12f is larger
# action scale is probably higher
# mass of host galaxy for normalization? Energy, action scales with mass of host galaxy

device = 'cpu'
sample_size = 1000
data_root = 'data/simulation'
dataset_name = 'm12i_cluster_data_large_cluster_v2'
dataset_path = os.path.join(data_root, dataset_name)

#23, 2, 26, 4, 1, 13, 21, 7, 5, 12, 14, 19
#
#22, 15, 18, 20, 3, 8
easy_small_clusters = [13, 21, 7, 5]
easy_mid_clusters = [6, 17, 14, 11, 8, 9, 1, 2]
easy_large_clusters = [22, 15, 18, 20, 3, 8]

def read_data():
    global sample_size, device, data_root, dataset_name, dataset_path
    df = pd.read_hdf(dataset_path+'.h5', key='star')

    df_std = pd.read_csv(dataset_path+'_std.csv')

    feature_columns = ['estar', 'feH', 'c_lzstar', 'jzstar', 'mgfe', 'vrstar', 'zstar', 'vphistar', 'eccstar']
    feature_weights = np.array([0.71216923,0.555757,0.31106377,0.1477975,0.13819067,0.1145066,0.08163675,0.07823427,0.07169756])
    feature_weights /= np.prod(feature_weights)

    sample_size = min(len(df), sample_size)
    sample_ids = np.random.choice(len(df), min(len(df), sample_size), replace=False)
    df_trim = df.iloc[sample_ids].copy()
    df = df_trim
    dataset = ClusterDataset(df_trim, feature_columns, 'cluster_id', feature_divs=df_std)
    #dataset.features *= np.array(feature_weights)[None,...]
    labels = dataset.labels.numpy()
    features = dataset.features.numpy()
    return features, labels



def evaluate_once(features, labels, n_components):
    #clusterer = C_HDBSCAN(metric='manhattan', min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_method='eom')
    clusterer = C_GaussianMixture(n_components=n_components)
    clusterer.add_data(features)
    clusters = clusterer.fit()

    cluster_eval = ClusterEvalAll(clusters, labels)
    print(cluster_eval())
    return cluster_eval()


# features, labels = read_data()
# evaluate_once(features, labels, 2,1)

# F1s = np.zeros((100))
# for i in range(6):
#   features, labels = read_data()
#   for n_components in range(1, 100):
#       results = evaluate_once(features, labels, n_components)
#       F1s[n_components] += results['Mode_recall']/6

# print(F1s)

# print(np.unravel_index(np.argmax(F1s), F1s.shape))
# print(np.max(F1s))
for n_components in [10,20,30,40,50,80,120,200]:
    t_results = []
    for i in range(60):
        features, labels = read_data()
        results = evaluate_once(features, labels, n_components)
        t_results.append(results)

    results = ClusterEvalAll.aggregate(t_results)
    print(results)
    with open(f'results/simple_gaussian_1000_{n_components}.json', 'w') as f:
        json.dump(results, f)




