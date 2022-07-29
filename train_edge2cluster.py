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

feature_columns = ['estar', 'feH', 'c_lzstar', 'jzstar', 'mgfe', 'vrstar', 'zstar', 'vphistar', 'eccstar']

def get_dataset(df, df_norm, sample_size, feature_columns):
    sample_size = min(len(df), sample_size)
    sample_ids = np.random.choice(len(df), min(len(df), sample_size), replace=False)
    df_trim = df.iloc[sample_ids].copy()
    dataset = GraphDataset(df_trim, feature_columns, 'cluster_id', 999, normalize=False, feature_norms=df_norm)
    dataset.initialize_dense(to_dense=False)
    return dataset

def compute_distance(dataset, model_name):
    with torch.no_grad():
        model = torch.load(os.path.join('weights',model_name),map_location=torch.device('cpu')).to(device)
        model.device = device
        model.config(False)
        model.eval()
        A, X, C = dataset[0]
        dA = A.coalesce().to_dense()
        D = dataset.D
        A,dA,X,C,D = A.to(device),dA.to(device),X.to(device),C.to(device),D.to(device)
        model.add_graph(D,dA,X)
        FX, SX = model(X)
        SX = SX.detach().to_sparse()
        FX = FX.detach()
        E = torch.sparse_coo_tensor(A.indices(), 1-SX.values(), A.shape).coalesce()
        D = torch.zeros((E.shape[0]))
        for i, v in zip(E.indices()[0], E.values()):
            D[i] += v
        for i, v in zip(E.indices()[1], E.values()):
            D[i] += v
        D /= 2
        weights = E.values() / torch.sqrt(D[E.indices()[0]] * D[E.indices()[1]])
        A = torch.sparse_coo_tensor(E.indices(), weights, E.shape).coalesce()
    return A, E, X

def train_epoch_step(epoch, A, E, X, model, optimizer, device):
    model.train()
    model.config(True)
    A,E,X = A.to(device),E.to(device),X.to(device)
    model.add_graph(A)
    model.add_connectivity(E.values())
    loss = model(X)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

def evaluate_step(epoch, A, E, X, labels, model, device):
    model.eval()
    model.config(False)
    A,E,X = A.to(device),E.to(device),X.to(device)
    model.add_graph(A)
    model.add_connectivity(E.values())
    FX = model(X).detach().numpy()
    metrics = ClusterEvalAll(FX, labels.numpy())
    print(f'metrics for epoch {epoch}:\n {metrics()}')
    return metrics()

model_name_simple = f'm12i_model_edgegen_32_32_epoch{1400}.pth'
model = GCNEdge2Cluster(len(feature_columns), num_cluster=30, graph_layer_sizes=[64], regularizer=0.00001, device=device)
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

t_results = []
for epoch in range(EPOCH):
    if epoch % 100 == 0:
        with torch.no_grad():
            if epoch != 0:
                results = evaluate_step(epoch, A, E, X, dataset.labels, model, device)
                t_results.append(results)
                if len(t_results) > 10:
                    t_results.pop(0)
                p_results = ClusterEvalAll.aggregate(t_results)
                print(p_results)
            dataset = get_dataset(df, df_norm, sample_size, feature_columns)
            A, E, X = compute_distance(dataset, model_name_simple)
    loss = train_epoch_step(epoch, A, E, X, model, optimizer, device)
    print(loss)


