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
sample_size = 10000
data_root = 'data/simulation'
dataset_name = 'm12i_cluster_data_large_cluster_v2'
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
    df_std = pd.read_csv(dataset_path+'_std.csv')


    feature_columns = ['estar', 'lzstar', 'lxstar', 'lystar', 'jzstar', 'jrstar', 'eccstar', 'rstar', 'feH', 'mgfe', 'xstar', 'ystar', 'zstar', 'vxstar', 'vystar', 'vzstar', 'vrstar', 'vphistar', 'vthetastar', 'omegaphistar', 'omegarstar', 'omegazstar', 'thetaphistar', 'thetarstar', 'thetazstar', 'zmaxstar']
    sample_size = min(len(df), sample_size)
    sample_ids = np.random.choice(len(df), min(len(df), sample_size), replace=False)
    df_trim = df.iloc[sample_ids].copy()
    df = df_trim
    dataset = GraphDataset(df_trim, feature_columns, 'cluster_id', 10, feature_divs=df_std)
    dataset.initialize()
    labels = dataset.labels

    with torch.no_grad():
        model = torch.load(os.path.join('weights',model_name)) # 229
        model.config(False)
        model.eval()
        A, X, C = dataset[0]
        model.add_graph(A.to(device))
        SX = model(X.to(device))
        dist = np.ones((len(X), len(X)))
        dist[A.indices()[0].numpy(), A.indices()[1].numpy()] = SX
        dist = np.minimum(dist, np.transpose(dist))

        clusterer = C_Spectral(n_components=n_components, assign_labels='kmeans')
        clusterer.add_data(1-dist)
        clusters = clusterer.fit()

    cluster_eval = ClusterEvalAll(clusters, labels.numpy())
    print(cluster_eval())
    return cluster_eval()


model_name_simple = f'model_gnn_edge_32_32__epoch{390}.pth'

for n_components in [30,40,50,80,120,200]:
    t_results = []
    for i in range(60):
        results = evaluate_once(model_name_simple, n_components)
        t_results.append(results)

    results = ClusterEvalAll.aggregate(t_results)
    print(results)
    with open(f'results/gnn_edge_spectral_1000_{n_components}.json', 'w') as f:
        json.dump(results, f)
