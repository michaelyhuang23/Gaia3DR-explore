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
from cluster_analysis import C_HDBSCAN,C_Spectral
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

    feature_columns = ['estar', 'lzstar', 'lxstar', 'lystar', 'jzstar', 'jrstar', 'eccstar', 'rstar', 'feH', 'mgfe', 'xstar', 'ystar', 'zstar', 'vxstar', 'vystar', 'vzstar', 'vrstar', 'vphistar', 'vthetastar', 'omegaphistar', 'omegarstar', 'omegazstar', 'thetaphistar', 'thetarstar', 'thetazstar', 'zmaxstar']
    sample_size = min(len(df), sample_size)
    sample_ids = np.random.choice(len(df), min(len(df), sample_size), replace=False)
    df_trim = df.iloc[sample_ids].copy()
    df = df_trim
    dataset = ClusterDataset(df_trim, feature_columns, 'cluster_id', feature_norms=df_norm)
    labels = dataset.labels

    def compute_distance(model, dataset, sample_size):
        B = len(dataset)
        L_features = dataset.features[None,...].repeat(B,1,1).reshape(B*B, dataset.features.shape[-1])
        R_features = dataset.features[:,None,:].repeat(1,B,1).reshape(B*B, dataset.features.shape[-1])
        dist = model(L_features, R_features).reshape(B,B)
        return dist

    colors = ['#%06X' % randint(0, 0xFFFFFF) for i in range(100)]


    with torch.no_grad():
        model = torch.load(os.path.join('weights',model_name)) # 229
        model.eval()
        model.config(False)

        dist = compute_distance(model, dataset, sample_size)
    #   dist[dist >= 0.7] = 1
        dist = np.minimum(dist, np.transpose(dist))

        #clusterer = C_HDBSCAN(metric='precomputed', min_cluster_size=2, min_samples=1, cluster_selection_method='leaf', cluster_selection_epsilon=0.01)
        clusterer = C_Spectral(n_components=n_components, assign_labels='kmeans')
        clusterer.add_data(1-dist)
        clusters = clusterer.fit()
        # x = clusterer.cluster.affinity_matrix_[np.nonzero(clusterer.cluster.affinity_matrix_)]
        # print(np.mean(x), np.std(x))

    cluster_eval = ClusterEvalAll(clusters, labels.numpy())
    print(cluster_eval())
    return cluster_eval()


model_name_complex = f'model_contrastive_64_64_epoch{25}.pth'
model_name_mid = f'model_contrastive_32_32_epoch{179}.pth'
model_name_simple = f'model_contrastive_scale_epoch{17}.pth'

# for n_components in [30,40,50,80,120,200]:
#     t_results = []
#     for i in range(60):
#         results = evaluate_once(model_name_simple, n_components)
#         t_results.append(results)

#     results = ClusterEvalAll.aggregate(t_results)
#     print(results)
#     with open(f'results/simple_spectral_1000_{n_components}.json', 'w') as f:
#         json.dump(results, f)



t_results = []
for i in range(60):
    results = evaluate_once(model_name_simple, 30)
    t_results.append(results)

results = ClusterEvalAll.aggregate(t_results)
print(results)













    # connectivity = UnionFind(dist.shape[0])
    # net = Network()
    # unique_labels = np.unique(dataset.labels)
    # for i,lab in enumerate(dataset.labels):
    #   lab_cluster = int(clusters[i])
    #   net.add_node(i, label=lab.item(), color=colors[lab.item()], title=str(lab.item()))
    # edges = []
    # for i in range(sample_size):
    #   for j in range(i+1, sample_size):
    #       if dist[i,j]<0.5:
    #           net.add_edge(i, j, weight=1-dist[i, j], value=1-dist[i, j], title=str(np.round(1-dist[i, j],1)))
    #       #edges.append((dist[i,j], i, j))
    # # edges.sort()
    # # for d, i, j in edges:
    # #     if connectivity.connect(i,j) : continue
    # #     connectivity.join(i,j)
    # #     net.add_edge(i, j, weight=1-dist[i, j], value=1-dist[i, j], title=str(np.round(1-dist[i, j],1)))
    # net.toggle_physics(False)
    # net.show('cluster_graph.html')
# cluster_names = [f'cluster_{label}' for label in clusters]
# fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# clusters_names = [f'cluster {label}' for label in clusters]
# sns.scatterplot(data=df, ax=axes[0,0], x='lzstar', y='estar', hue=clusters_names)
# sns.scatterplot(data=df, ax=axes[0,1], x='lystar', y='estar', hue=clusters_names)
# sns.scatterplot(data=df, ax=axes[0,2], x='rstar', y='eccstar', hue=clusters_names)
# sns.scatterplot(data=df, ax=axes[1,0], x='jphistar', y='jzstar', hue=clusters_names)
# sns.scatterplot(data=df, ax=axes[1,1], x='jphistar', y='jrstar', hue=clusters_names)
# sns.scatterplot(data=df, ax=axes[1,2], x='vxstar', y='vphistar', hue=clusters_names)
# plt.show()
