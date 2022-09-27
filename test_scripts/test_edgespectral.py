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
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter
import sklearn
import json

from neural_dataset import *
from gnn_cluster import *
from cluster_analysis import C_HDBSCAN,C_Spectral
from evaluation_metrics import *
from utils import cart2spherical, UnionFind


device = 'cpu'
sample_size = 1000
data_root = 'data/simulation'
dataset_name = 'm12f_cluster_data_large_cluster_v2'
dataset_path = os.path.join(data_root, dataset_name)
EPOCH = 40000

df = pd.read_hdf(dataset_path+'.h5', key='star')
with open(dataset_path+'_norm.json', 'r') as f:
    df_norm = json.load(f)
df_norm['mean']['lzstar'] = 0
df_norm['mean']['lxstar'] = 0
df_norm['mean']['lystar'] = 0
df_norm['mean']['jzstar'] = 0
df_norm['mean']['jrstar'] = 0

# feature_columns = ['estar', 'feH', 'c_lzstar', 'jzstar', 'mgfe', 'vrstar', 'zstar', 'vphistar', 'eccstar']
feature_columns = ['estar', 'feH', 'lzstar', 'lystar', 'lxstar', 'jzstar', 'jrstar', 'mgfe','eccstar', 'zstar']

def get_dataset(df, df_norm, sample_size, feature_columns):
    sample_size = min(len(df), sample_size)
    sample_ids = np.random.choice(len(df), min(len(df), sample_size), replace=False)
    df_trim = df.iloc[sample_ids].copy()
    dataset = GraphDataset(df_trim, feature_columns, 'cluster_id', 999, normalize=False, feature_norms=df_norm)
    dataset.initialize_dense()
    return dataset

def compute_distance(dataset, model_name):
    with torch.no_grad():
        model = torch.load(os.path.join('weights',model_name), map_location='cpu')
        model.device = device
        model.config(False)
        model.eval()
        A, X, C = dataset[0]
        D = dataset.D
        A,X,C,D = A.to(device),X.to(device),C.to(device),D.to(device)
        n = X.shape[0]
        C = dataset.labels[None,...].repeat(n, 1) != dataset.labels[...,None].repeat(1, n)
        model.add_graph(D,A,X)
        SX = model(X)
        SX = SX.detach()
        print(SX.shape,C.shape)
        preds = np.rint(SX.numpy()).astype(np.int32).flatten()
        class_metrics = ClassificationAcc(preds, C.numpy().astype(np.int32).flatten(), 2)
        print(f'test acc: {class_metrics.precision}\n{class_metrics.count_matrix}')
        SX = torch.sparse_coo_tensor(A.indices(), SX, A.shape).to_dense()
        E = (1-SX)
        # E = torch.sparse_coo_tensor(A.indices(), 1-SX, A.shape)
    return E


model_name_simple = f'm12i_dense_small_model_32_32_epoch{1000}.pth'

t_results = []
for i in range(10):
    dataset = get_dataset(df, df_norm, sample_size, feature_columns)
    E = compute_distance(dataset, model_name_simple)

    clusterer = C_Spectral(n_components=30, assign_labels='kmeans')
    dist = E.to_dense().numpy()
    dist = (dist + np.transpose(dist))/2
    dist += 0.03
    np.fill_diagonal(dist, 0)
    print(np.mean(dist), np.std(dist))
    clusterer.add_data(dist)
    clusters = clusterer.fit()

    metrics = ClusterEvalAll(clusters, dataset.labels.numpy())
    print(metrics())
    t_results.append(metrics())

results = ClusterEvalAll.aggregate(t_results)
print(results)



