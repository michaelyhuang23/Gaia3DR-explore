import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AffinityPropagation
from sklearn.cluster.mixture import GaussianMixture
from hdbscan import HDBSCAN

class Clusterer:
    def __init__(self):
        super().__init__()

    def config(self, cluster_info=None):
        pass

    def add_data(self, data, **kwargs):
        pass

    def fit(self):
        pass

class C_HDBSCAN(Clusterer):
    def __init__(self, metric='euclidean', min_cluster_size=5, min_samples=None, alpha=1.0):
        super().__init__()
        self.cluster = HDBSCAN(algorithm='best', alpha=alpha, approx_min_span_tree=True,
    gen_min_span_tree=False, leaf_size=40, metric=metric, min_cluster_size=min_cluster_size, min_samples=min_samples, p=None)

    def config(self, cluster_info=None):
        pass

    def add_data(self, data, columns = None):
        if columns != None:
            self.data = data[columns].copy().to_numpy()
        else:
            self.data = data.copy().to_numpy()

    def fit(self):
        self.cluster.fit(self.data)
        return self.cluster.labels_
        
