import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AffinityPropagation
from sklearn.mixture import GaussianMixture
from hdbscan import HDBSCAN
from scipy import stats
from collections import Counter

class Clusterer:
    def __init__(self):
        super().__init__()

    def config(self, cluster_info=None):
        pass

    def add_data(self, standardization_method, **kwargs):
        if standardization_method == 'std':
            self.data -= np.mean(self.data, axis=0)[None,...]
            self.data /= np.std(self.data, axis=0)[None,...]

    def fit(self):
        pass

class C_HDBSCAN(Clusterer):
    def __init__(self, metric='euclidean', min_cluster_size=5, min_samples=None, alpha=1.0, allow_single_cluster=False, cluster_selection_method='eom', cluster_selection_epsilon=0):
        super().__init__()
        self.cluster = HDBSCAN(algorithm='best', alpha=alpha, approx_min_span_tree=True,
    gen_min_span_tree=False, leaf_size=40, metric=metric, min_cluster_size=min_cluster_size, 
    min_samples=min_samples, allow_single_cluster=allow_single_cluster, p=None, cluster_selection_method=cluster_selection_method,
    cluster_selection_epsilon=cluster_selection_epsilon)

    def config(self, cluster_info=None):
        pass

    def add_data(self, data, columns=None, standardization_method='std'):
        if columns != None:
            self.data = data[columns].copy().to_numpy()
        else:
            self.data = data.copy().to_numpy()
        super().add_data(standardization_method=standardization_method)

    def fit(self):
        self.cluster.fit(self.data)
        return self.cluster.labels_



class C_GaussianMixture(Clusterer):
    def __init__(self, n_components=1, tol=0.001, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None):
        super().__init__()
        self.cluster = GaussianMixture(n_components=n_components, covariance_type='full', tol=tol, reg_covar=1e-06, max_iter=100, n_init=1, init_params=init_params, weights_init=weights_init, means_init=means_init, precisions_init=precisions_init)

    def config(self, cluster_info=None):
        pass

    def add_data(self, data, columns=None, standardization_method='std'):
        if columns != None:
            self.data = data[columns].copy().to_numpy()
        else:
            self.data = data.copy().to_numpy()
        super().add_data(standardization_method=standardization_method)

    def fit(self, epoch=100):
        self.cluster.max_iter = epoch
        return self.cluster.fit_predict(self.data)
        


class ClusterEvalIoU:
    def __init__(self, preds, labels, IoU_thres=0.5):
        super().__init__()
        self.preds = preds
        self.labels = labels
        self.IoU_thres = IoU_thres

        unique_labels = Counter(list(self.labels))
        unique_preds = Counter(list(self.preds))

        self.TP = 0
        for cluster_id in unique_labels.keys():
            point_ids = np.argwhere(self.labels == cluster_id)[:,0]
            mode, count = stats.mode(self.preds[point_ids], axis=None)
            mode = mode[0]
            count = count[0]
            #print(count, unique_labels[cluster_id], unique_preds[mode])
            IoU = count / (unique_labels[cluster_id]+unique_preds[mode]-count)
            if mode>-1 and IoU >= IoU_thres:
                self.TP+=1

        self.P = len(unique_preds) - (1 if unique_preds[-1]!=0 else 0)
        self.T = len(unique_labels)
        self.precision = self.TP / self.P  # what percent of clusters are actual clusters
        self.recall = self.TP / self.T  # what percent of actual clusters are identified

    def __call__(self):
        return self.precision, self.recall

