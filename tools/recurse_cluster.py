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
        self.clusterer1.config(cluster_info)
        
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
        cluster_idff = np.zeros((len(self.data1)), dtype=np.int32)
        for i in range(self.max_epoch if epoch==None else epoch):
            c_count = 0
            cluster_id1 = np.ones((len(self.data1)), dtype=np.int32)*(-1)
            cluster_id2 = np.ones((len(self.data1)), dtype=np.int32)*(-1)

            print(cluster_idff)
            
            for j in range(0, np.max(cluster_idff)+1):
                idx = np.argwhere(cluster_idff == j)[:,0]
                self.clusterer1.add_data(self.data1.iloc[idx].copy())
                cluster_id1_t = self.clusterer1.fit()
                increment = np.max(cluster_id1_t)+1
                if increment == 0:
                    cluster_id1_t += 1
                    increment = 1
                cluster_id1_t = np.where(cluster_id1_t>-1, cluster_id1_t+c_count, -1)
                cluster_id1[idx] = cluster_id1_t
                c_count += increment

            print(cluster_id1)
            print(c_count)
            from collections import Counter
            cc = Counter(cluster_id1)
            for i in range(0,c_count):
                assert cc[i]>0

            c_count = 0
            for j in range(0, np.max(cluster_id1)+1):
                idx = np.argwhere(cluster_id1 == j)[:,0]
                self.clusterer2.add_data(self.data2.iloc[idx].copy())
                cluster_id2_t = self.clusterer2.fit()
                increment = np.max(cluster_id2_t)+1
                if increment == 0:
                    cluster_id2_t += 1
                    increment = 1
                cluster_id2_t = np.where(cluster_id2_t>-1, cluster_id2_t+c_count, -1)
                cluster_id2[idx] = cluster_id2_t
                c_count += increment
                if c_count>1e9: 
                    raise AssertionError

            cc = Counter(cluster_id2)
            for i in range(0,c_count):
                assert cc[i]>0
            print(c_count)

            if np.all(cluster_idff == cluster_id2):
                break
            cluster_idff = cluster_id2

        return cluster_idff


# import pandas as pd
# import seaborn as sns
# import numpy as np
# from cluster_analysis import C_HDBSCAN

# df = pd.read_hdf('./data/m12i_cluster_data_large_mass_large_cluster.h5', key='star')

# clustera = C_HDBSCAN(metric='manhattan', min_cluster_size=2, min_samples=1)
# clusterb = C_HDBSCAN(metric='manhattan', min_cluster_size=2, min_samples=1)
# cluster2 = RecursiveCluster(subspace1_clusterer=clustera, subspace2_clusterer=clusterb)
# cluster2.config()

# cluster2.add_data(df, subspace1=['feH'], subspace2=['mgfe'])

# labels2 = cluster2.fit()
    
        
