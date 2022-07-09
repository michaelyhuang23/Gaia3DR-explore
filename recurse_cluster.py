import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from cluster_analysis import Clusterer, C_HDBSCAN

# all clustering models from cluster_analysis should have the same api

class RecursiveCluster(Clusterer):
    def __init__(self, subspace1_clusterer=None, subspace2_clusterer=None):
        super().__init__()
        assert subspace1_clusterer != None
        assert subspace2_clusterer != None
        self.clusterer1 = subspace1_clusterer
        self.clusterer2 = subspace2_clusterer
        self.max_epoch = 20

    def config(self, cluster_info=None):
        '''
        reinitialize with initial clusters, every clusterer should have this
        cluster_info is a dict containing keys like
        'cluster_ids' which is a list of ids of each point's cluster
        'cluster_centers' if the clustering algorithm has defined cluster_centers
        '''
        self.clusterer1.config(cluster_info):
        
    def add_data(self, data, subspace1=None, subspace2=None):
        if subspace1 != None:
            self.data1 = data[subspace1].copy()
        else:
            self.data1 = data.copy()

        if subspace2 != None:
            self.data2 = data[subspace2].copy()
        else:
            self.data2 = data.copy()

        self.clusterer1.add_data(self.data1)
        self.clusterer2.add_data(self.data2)

    def fit(self, epoch=None): 
        '''
        this is the code that actually performs the clustering. We always perform clustering on cluster1 first. We fit until convergence if epoch==None
        '''
        cluster_idff = np.ones((len(self.data1)), dtype=np.int32)*(-1)
        for i in range(self.max_epoch if epoch==None else epoch):
            c_count = 0
            cluster_id1 = np.ones((len(self.data1)), dtype=np.int32)*(-1)
            cluster_id2 = np.ones((len(self.data1)), dtype=np.int32)*(-1)
            
            for j in range(-1, np.max(cluster_idff)+1):
                idx = np.argwhere(cluster_idff == j)
                self.clusterer1.add_data(self.data1.loc[idx].copy())
                cluster_id1_t = self.clusterer1.fit()
                cluster_id1_t = cluster_id1_t.where(cluster_id1_t>-1, cluster_id1_t+c_count, -1)
                cluster_id1[idx] = cluster_id1_t
                c_count+=np.max(cluster_id1_t)+1

            c_count = 0
            for j in range(-1, np.max(cluster_id1)+1):
                idx = np.argwhere(cluster_id1 == j)
                self.clusterer2.add_data(self.data2.loc[idx].copy())
                cluster_id2_t = self.clusterer2.fit()
                cluster_id2_t = cluster_id2_t.where(cluster_id2_t>-1, cluster_id2_t+c_count, -1)
                cluster_id2[idx] = cluster_id2_t
                c_count+=np.max(cluster_id2_t)+1
            if cluster_idff == cluster_id2:
                break
            cluster_idff = cluster_id2

        return cluster_idff
    
        
        
