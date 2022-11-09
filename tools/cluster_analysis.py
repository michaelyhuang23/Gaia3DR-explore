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
from tools.evaluation_metrics import *

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
    def __init__(self, egnn_input_size, n_components, similar_weight=1, egnn_lr=0.01, egnn_regularizer=0.1, clustergen_lr=0.01, clustergen_regularizer=0.00001, device='cpu'):
        self.device = device
        self.n_components = n_components
        self.egnn_input_size = egnn_input_size
        self.egnn = GCNEdgeBased(egnn_input_size, similar_weight, regularizer=egnn_regularizer, device=self.device).to(self.device)
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

    def fit(self, EPOCH=4000):
        self.initialize_model()
        self.clustergen.train()
        self.clustergen.config(True)
        with torch.no_grad():
            self.egnn.config(False)
            SX, egnn_loss = self.egnn(self.X)
            SX, egnn_loss = SX.detach(), egnn_loss.detach()
            preds = np.rint(SX.cpu().numpy()).astype(np.int32)
            metrics = ClassificationAcc(preds, self.C.cpu().numpy().astype(np.int32), 2)
            print(f'test acc: {metrics.precision}\n{metrics.count_matrix}')
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
            if (epoch+1)%100 == 0:
                print(loss.item())

        self.clustergen.eval()
        self.clustergen.config(False)
        self.clustergen.add_graph(self.AE)
        self.clustergen.add_connectivity(self.E.values())
        FX = self.clustergen(self.X).detach().cpu().numpy()
        return FX, egnn_loss

    def save_model(self, root, epoch):
        egnn_path = os.path.join(root, 'egnn')
        clustergen_path = os.path.join(root, 'clustergen')
        torch.save(self.egnn, os.path.join(egnn_path, f'epoch{epoch}.pth'))
        torch.save(self.clustergen, os.path.join(clustergen_path, f'epoch{epoch}.pth'))

    def load_model(self, root, epoch):
        egnn_path = os.path.join(root, 'egnn')
        self.egnn = torch.load(os.path.join(egnn_path, f'epoch{epoch}.pth'), map_location=self.device)


class C_GNN_GMM(TrainableClusterer):
    def __init__(self, input_size, num_cluster, n_projection_dim, gnn_lr, device='cpu'):
        self.device = device
        self.num_cluster = num_cluster
        self.model = GNNProjection(input_size, n_projection_dim, graph_layer_sizes=[32], regularizer=None, device=self.device)
        self.model_optim = Adam(self.model.parameters(), lr=gnn_lr, weight_decay=1e-5)

    def add_data(self, dataset):
        self.X, self.A, self.labels = dataset.X.to(self.device), dataset.A.to(self.device), dataset.labels.to(self.device)
        self.model.add_graph(self.A)
        self.cluster_labels = F.one_hot(self.labels)
        self.label_names = [str(lab.item()) for lab in self.labels]

    def gmm_predict_proba(self, gmm_result, X, n_components):
        dim = X.shape[-1]
        costs = torch.zeros((n_components, X.shape[0]),device=self.device)
        for i in range(n_components):
            pos = X - torch.tensor(gmm_result.means_[i], device=self.device)
            cov = torch.tensor(gmm_result.covariances_[i], device=self.device)
            prec = torch.tensor(gmm_result.precisions_[i], device=self.device)
            costs[i] = gmm_result.weights_[i]*1/torch.linalg.det(cov)**0.5*1/(2*3.1415)**(dim/2) * torch.exp(-0.5* torch.sum(torch.mm(pos, prec) * pos, dim=-1))
        costs = torch.transpose(costs, 0, 1)
        return costs / torch.sum(costs, dim=-1)[...,None]

    def compute_loss(self, X, return_result=False):
        if return_result is True:
            n_components = self.num_cluster
        else:
            n_components = self.cluster_labels.shape[-1]
        with torch.no_grad():
            gmm_result = GaussianMixture(n_components=n_components).fit(X.detach().cpu().numpy())
        FX = self.gmm_predict_proba(gmm_result, X, n_components)  
        SX = torch.log(torch.clamp(FX,min=0.001))
        corr = -torch.mm(torch.transpose(self.cluster_labels.float(),0,1), SX)
        with torch.no_grad():
            label_idx, pred_idx = linear_sum_assignment(corr.detach().cpu().numpy())
        loss = torch.mean(corr[label_idx, pred_idx])
        if return_result:
            return FX, loss
        else:
            return loss

    def train(self):
        self.model.config(True)
        self.model.train()
        p_X = self.model(self.X)
        loss = self.compute_loss(p_X, False)
        loss.backward()
        self.model_optim.step()
        self.model_optim.zero_grad()
        self.loss = loss.item()
        return self.loss

    def fit(self, EPOCH=2000):
        with torch.no_grad():
            self.model.config(False)
            self.model.eval()
            p_X = self.model(self.X)
            X_out = p_X.detach().cpu().numpy()
            fig = px.scatter_3d(x=X_out[:,0], y=X_out[:,1], z=X_out[:,2],color=self.label_names, opacity=1, size_max=10)
            fig.show()
            SX, loss = self.compute_loss(p_X, True)
            SX, loss = SX.detach(), loss.detach()
            fig2 = px.scatter_3d(x=X_out[:,0], y=X_out[:,1], z=X_out[:,2],color=torch.argmax(SX,dim=-1).cpu().long().numpy(), opacity=1, size_max=10)
            fig2.show()
        return SX, loss

    def save_model(self, root, epoch):
        model_path = os.path.join(root, 'model')
        torch.save(self.model, os.path.join(model_path, f'epoch{epoch}.pth'))

    def load_model(self, root, epoch):
        model_path = os.path.join(root, 'model')
        self.model = torch.load(os.path.join(model_path, f'epoch{epoch}.pth'), map_location=self.device)


class C_GNN_KMeans(TrainableClusterer):
    def __init__(self, input_size, num_cluster, n_projection_dim, gnn_lr, device='cpu'):
        self.device = device
        self.num_cluster = num_cluster
        self.model = GNNProjection(input_size, n_projection_dim, graph_layer_sizes=[32], regularizer=None, device=self.device)
        self.model_optim = Adam(self.model.parameters(), lr=gnn_lr, weight_decay=1e-5)

    def add_data(self, dataset):
        self.X, self.A, self.labels = dataset.X.to(self.device), dataset.A.to(self.device), dataset.labels.to(self.device)
        self.model.add_graph(self.A)
        self.cluster_labels = F.one_hot(self.labels)
        self.label_names = [str(lab.item()) for lab in self.labels]

    def kmeans_compute_dist(self, kmeans_result, X):
        labels = torch.tensor(kmeans_result.labels_, device=self.device).long()
        centers = torch.tensor(kmeans_result.cluster_centers_, device=self.device)
        other_costs = torch.exp(torch.mean(torch.log(torch.clamp(torch.norm(X[:,None,:] - centers[None,:,:], dim=-1), min=0.001)), dim=-1))
        costs = torch.norm(X - centers[labels], dim=-1) / other_costs
        return costs

    def compute_loss(self, X, return_result=False):
        if return_result is True:
            n_components = self.num_cluster
        else:
            n_components = self.cluster_labels.shape[-1]
        with torch.no_grad():
            kmeans_result = KMeans(n_clusters=n_components).fit(X.detach().cpu().numpy())
        LX = self.kmeans_compute_dist(kmeans_result, X)
        print(LX.shape)
        loss = torch.mean(LX)
        if return_result:
            return torch.tensor(kmeans_result.labels_, device=self.device).long(), loss
        else:
            return loss

    def train(self):
        self.model.config(True)
        self.model.train()
        p_X = self.model(self.X)
        print(torch.mean(p_X), torch.std(p_X))
        loss = self.compute_loss(p_X, False)
        loss.backward()
        self.model_optim.step()
        self.model_optim.zero_grad()
        self.loss = loss.item()
        return self.loss

    def fit(self, EPOCH=2000):
        with torch.no_grad():
            self.model.config(False)
            self.model.eval()
            p_X = self.model(self.X)
            X_out = p_X.detach().cpu().numpy()
            fig = px.scatter_3d(x=X_out[:,0], y=X_out[:,1], z=X_out[:,2],color=self.label_names, opacity=1, size_max=10)
            fig.show()
            SX, loss = self.compute_loss(p_X, True)
            SX, loss = SX.detach(), loss.detach()
        return SX, loss

    def save_model(self, root, epoch):
        model_path = os.path.join(root, 'model')
        torch.save(self.model, os.path.join(model_path, f'epoch{epoch}.pth'))

    def load_model(self, root, epoch):
        model_path = os.path.join(root, 'model')
        self.model = torch.load(os.path.join(model_path, f'epoch{epoch}.pth'), map_location=self.device)
