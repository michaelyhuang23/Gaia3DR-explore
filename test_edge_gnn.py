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

from neural_dataset import *
from cluster_analysis import C_HDBSCAN,C_Spectral
from evaluation_metrics import ClusterEvalAll
from utils import cart2spherical, UnionFind

# apply on m12f, scale of m12f is larger
# action scale is probably higher
# mass of host galaxy for normalization? Energy, action scales with mass of host galaxy

device = 'cpu'
sample_size = 1000
data_root = 'data/simulation'
dataset_name = 'm12f_cluster_data_large_cluster_v2'
dataset_path = os.path.join(data_root, dataset_name)

#23, 2, 26, 4, 1, 13, 21, 7, 5, 12, 14, 19
#
#22, 15, 18, 20, 3, 8
easy_small_clusters = [13, 21, 7, 5]
easy_mid_clusters = [6, 17, 14, 11, 8, 9, 1, 2]
easy_large_clusters = [22, 15, 18, 20, 3, 8]


def evaluate_once(model_name, n_components):
    global device, sample_size, data_root, dataset_name, dataset_path, easy_large_clusters, easy_mid_clusters, easy_small_clusters
    df = pd.read_hdf(dataset_path+'.h5', key='star')
    with open(dataset_path+'_norm.json', 'r') as f:
        df_norm = json.load(f)

    df_norm['mean']['lzstar'] = 0
    df_norm['mean']['lxstar'] = 0
    df_norm['mean']['lystar'] = 0
    df_norm['mean']['jzstar'] = 0
    df_norm['mean']['jrstar'] = 0

    #feature_columns = ['estar', 'lzstar', 'lxstar', 'lystar', 'jzstar', 'jrstar', 'eccstar', 'rstar', 'feH', 'mgfe', 'xstar', 'ystar', 'zstar', 'vxstar', 'vystar', 'vzstar', 'vrstar', 'vphistar', 'vthetastar', 'omegaphistar', 'omegarstar', 'omegazstar', 'thetaphistar', 'thetarstar', 'thetazstar', 'zmaxstar']
    feature_columns = ['estar', 'feH', 'c_lzstar', 'jzstar', 'mgfe', 'vrstar', 'zstar', 'vphistar', 'eccstar']
    sample_size = min(len(df), sample_size)
    sample_ids = np.random.choice(len(df), min(len(df), sample_size), replace=False)
    df_trim = df.iloc[sample_ids].copy()
    df = df_trim
    dataset = GraphDataset(df_trim, feature_columns, 'cluster_id', 999, normalize=False, feature_norms=df_norm)
    dataset.initialize()
    labels = dataset.labels

    with torch.no_grad():
        model = torch.load(os.path.join('weights',model_name)) # 229
        model.config(False)
        model.eval()
        A, X, C = dataset[0]
        D = dataset.D
        A,X,C,D = A.to(device),X.to(device),C.to(device),D.to(device)
        model.add_graph(D,A,X)
        SX = model(X)
        dist = np.ones((len(X), len(X)))
        dist[A.indices()[0].numpy(), A.indices()[1].numpy()] = SX
        dist = (dist + np.transpose(dist))/2

        clusterer = C_Spectral(n_components=n_components, assign_labels='kmeans', affinity='precomputed')
        clusterer.add_data(1-dist)
        clusters = clusterer.fit()
        x = clusterer.cluster.affinity_matrix_[np.nonzero(clusterer.cluster.affinity_matrix_)]
        print(np.mean(x), np.std(x))

    cluster_eval = ClusterEvalAll(clusters, labels.numpy())
    print(cluster_eval())
    return cluster_eval()


model_name_simple = f'm12i_model_32_32_epoch{400}.pth'

# for n_components in [10,20,30,40,50,80,120,200]:
#     t_results = []
#     for i in range(10):
#         results = evaluate_once(model_name_simple, n_components)
#         t_results.append(results)

#     results = ClusterEvalAll.aggregate(t_results)
#     print(results)
#     with open(f'results/gnn_edge_spectral_1000_{n_components}.json', 'w') as f:
#         json.dump(results, f)

t_results = []
for i in range(10):
    results = evaluate_once(model_name_simple, 30)
    t_results.append(results)

results = ClusterEvalAll.aggregate(t_results)
print(results)
