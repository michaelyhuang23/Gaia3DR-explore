import pandas as pd
import numpy as np
import os
import copy
from sklearn.cluster import DBSCAN, KMeans, AffinityPropagation, SpectralClustering
from sklearn.mixture import GaussianMixture
from hdbscan import HDBSCAN
from scipy import stats
from collections import Counter

import torch
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter

from tools.gnn_cluster import *

class Clusterer:
    def __init__(self):
        super().__init__()

    def add_data(self, data):
        pass

    def fit(self):
        pass

class C_Spectral(Clusterer):
    def __init__(self, n_components=8, affinity='precomputed', assign_labels='discretize', n_neighbors=10):
        super().__init__()
        self.cluster = SpectralClustering(n_components, affinity=affinity, assign_labels=assign_labels, n_neighbors=n_neighbors)

    def add_data(self, adj):
        self.affinity_matrix = adj

    def fit(self):
        self.cluster.fit(self.affinity_matrix)
        return self.cluster.labels_

class C_HDBSCAN(Clusterer):
    def __init__(self, metric='euclidean', min_cluster_size=5, min_samples=None, alpha=1.0, allow_single_cluster=False, cluster_selection_method='eom', cluster_selection_epsilon=0):
        super().__init__()
        self.cluster = HDBSCAN(algorithm='best', alpha=alpha, approx_min_span_tree=True,
    gen_min_span_tree=False, leaf_size=40, metric=metric, min_cluster_size=min_cluster_size, 
    min_samples=min_samples, allow_single_cluster=allow_single_cluster, p=None, cluster_selection_method=cluster_selection_method,
    cluster_selection_epsilon=cluster_selection_epsilon)

    def add_data(self, dataset):
        self.data = dataset.features

    def fit(self):
        self.cluster.fit(self.data)
        return self.cluster.labels_

class C_GaussianMixture(Clusterer):
    def __init__(self, n_components=1, tol=0.001, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None):
        super().__init__()
        self.cluster = GaussianMixture(n_components=n_components, covariance_type='full', tol=tol, reg_covar=1e-06, max_iter=100, n_init=1, init_params=init_params, weights_init=weights_init, means_init=means_init, precisions_init=precisions_init)

    def add_data(self, dataset):
        self.data = dataset.features

    def fit(self, epoch=100):
        self.cluster.max_iter = epoch
        return self.cluster.fit_predict(self.data)
        


class TrainableClusterer(Clusterer):
    def __init__(self):
        super().__init__()

    def add_data(self, data):
        pass

    def fit(self):
        pass

    def train(self):
        pass


class C_SNC(TrainableClusterer):
    def __init__(self, egnn_input_size, n_components, similar_weight=1, egnn_lr=0.01, clustergen_lr=0.01, clustergen_regularizer=0.00001, device='cpu'):
        self.device = device
        self.n_components = n_components
        self.egnn_input_size = egnn_input_size
        self.egnn = GCNEdgeBased(egnn_input_size, similar_weight, self.device).to(self.device)
        self.egnn_optim = Adam(self.egnn.parameters(), lr=egnn_lr, weight_decay=1e-5)
        self.clustergen_lr = clustergen_lr
        self.clustergen_regularizer = clustergen_regularizer

    def add_data(self, dataset):
        self.D, self.A, self.X, self.C = dataset.D.to(self.device), dataset.A.to(self.device), dataset.X.to(self.device), dataset.C.to(self.device)
        self.egnn.add_graph(self.D,self.A,self.X)
        self.egnn.add_connectivity(self.C)

    def initialize_model(self):
        self.clustergen = GCNEdge2Cluster(self.egnn_input_size, num_cluster=self.n_components, graph_layer_sizes=[64], regularizer=self.clustergen_regularizer, device=self.device)
        self.clustergen_optim = Adam(self.clustergen.parameters(), lr=self.clustergen_lr, weight_decay=1e-5)

    def train(self):
        self.egnn.config(True)
        loss = self.egnn(self.X)
        loss.backward()
        self.egnn_optim.step()
        self.egnn_optim.zero_grad()
        self.loss = loss.item()
        return self.loss

    def fit(self, EPOCH=200):
        self.initialize_model()
        self.clustergen.train()
        self.clustergen.config(True)
        with torch.no_grad():
            self.egnn.config(False)
            SX = self.egnn(self.X).detach()
            self.E = torch.sparse_coo_tensor(self.A.indices(), SX, self.A.shape).coalesce() # E denotes affinity
            self.DE = (torch.sparse.sum(self.E, dim=0).coalesce().to_dense() + torch.sparse.sum(self.E, dim=1).coalesce().to_dense())/2 # computes affinity "degree"
            weights = self.E.values() / torch.sqrt(self.DE[self.E.indices()[0]] * self.DE[self.E.indices()[1]])
            self.AE = torch.sparse_coo_tensor(self.E.indices(), weights, self.E.shape).coalesce() # AE is the normalized affinity

        for epoch in range(EPOCH):
            self.clustergen.add_graph(self.AE)
            self.clustergen.add_connectivity(self.E.values())
            loss = self.clustergen(self.X)
            loss.backward()
            self.clustergen_optim.step()
            self.clustergen_optim.zero_grad()
            if (epoch+1)%10 == 0:
                print(loss.item())

        self.clustergen.eval()
        self.clustergen.config(False)
        self.clustergen.add_graph(self.AE)
        self.clustergen.add_connectivity(self.E.values())
        FX = self.clustergen(self.X).detach().numpy()
        return FX

    def save_model(self, root, epoch):
        egnn_path = os.path.join(root, 'egnn')
        clustergen_path = os.path.join(root, 'clustergen')
        torch.save(self.egnn, os.path.join(egnn_path, f'epoch{epoch}.pth'))
        torch.save(self.clustergen, os.path.join(clustergen_path, f'epoch{epoch}.pth'))



