from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
import torch

def sample_space(df_, radius=5, radius_sun=8, sample_size=1000):
    df = df_.copy()
    phi = np.random.uniform(0, np.pi*2)
    xsun = np.cos(phi)*radius_sun
    ysun = np.sin(phi)*radius_sun
    zsun = np.random.normal(0, 0.016)
    df = df.loc[(df['xstar'].to_numpy()-xsun)**2 + (df['ystar'].to_numpy()-ysun)**2 + (df['zstar'].to_numpy()-zsun)**2 < radius**2]
    if len(df) > sample_size:
        sample_ids = np.random.choice(len(df), min(len(df), sample_size), replace=False)
        df = df.iloc[sample_ids].copy()
    return df

class GlobalJitterTransform:
    def __init__(self, jitter_range=0.1):
        self.jitter_range = jitter_range
    def __call__(self, X):
        return X + torch.ones_like(X) * (random.random()*2-1) * self.jitter_range

class GlobalScaleTransform:
    def __init__(self, scale_range=0.1):
        self.scale_range = scale_range
    def __call__(self, X):
        return X * ((random.random()*2-1)*self.scale_range+1)

class JitterTransform:
    def __init__(self, jitter_range=0.1):
        self.jitter_range = jitter_range
    def __call__(self, X):
        return X + torch.randn_like(X) * self.jitter_range

class ScaleTransform:
    def __init__(self, scale_range=0.1):
        self.scale_range = scale_range
    def __call__(self, X):
        return X * (torch.randn_like(X) * self.scale_range + 1)

class ClusterDataset(Dataset):
    def __init__(self, dataframe, features, cluster_ids=None, feature_norms=None, scales=None):
        super().__init__() 
        self.features_subs = torch.tensor([feature_norms['mean'][feature] for feature in features])
        self.feature_divs = torch.tensor([feature_norms['std'][feature] for feature in features])
        if isinstance(features, np.ndarray):
            self.features = torch.tensor(features).float()
        else:
            self.features = torch.tensor(dataframe[features].to_numpy()).float()
        self.features -= self.features_subs[None,...]
        self.features /= self.feature_divs[None,...]

        if isinstance(scales, np.ndarray):
            self.features *= torch.tensor(scales)[None,...]

        if cluster_ids is None:
            self.labels = None
        elif isinstance(cluster_ids, str):
            self.labels = torch.tensor(dataframe[cluster_ids].to_numpy()).long()
        else:
            self.labels = torch.tensor(cluster_ids).long()
        if self.labels is not None:
            self.labels -= torch.min(self.labels)

        self.cluster_ids = list(set([label.item() for label in self.labels]))
        self.clusters = {}
        for i in range(len(self.labels)):
            if self.labels[i].item() not in self.clusters:
                self.clusters[self.labels[i].item()] = []
            self.clusters[self.labels[i].item()].append(i)

        assert self.labels is None or len(self.labels) == self.features.shape[0]

    def cluster_transform(self, transforms=None):
        if transforms is None:
            transforms = self.transforms
        for cluster_id in self.cluster_ids:
            features = self.features[torch.tensor(self.clusters[cluster_id])]
            for transform in transforms:
                features = transform(features)
            self.features[torch.tensor(self.clusters[cluster_id])] = features

    def global_transform(self, transforms=None):
        if transforms is None:
            transforms = self.transforms
        for transform in transforms:
            self.features = transform(self.features)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        feature = self.features[idx]
        if self.labels is None:
            return feature, None
        else:
            return feature, self.labels[idx]

class ContrastDataset(ClusterDataset):
    def __init__(self, dataframe, features, cluster_ids, feature_norms=None, positive_percent=None, transforms=[]):
        super().__init__(dataframe, features, cluster_ids, feature_norms)
        if positive_percent is None:
            self.positive_percent = 0.2
        else:
            self.positive_percent = positive_percent
        
        self.cluster_ids = list(set([label.item() for label in self.labels]))
        self.clusters = {}
        for i in range(len(self.labels)):
            if self.labels[i].item() not in self.clusters:
                self.clusters[self.labels[i].item()] = []
            self.clusters[self.labels[i].item()].append(i)
        self.transforms = transforms
        assert self.labels is None or len(self.labels) == self.features.shape[0]

    def cluster_transform(self, transforms=None):
        if transforms is None:
            transforms = self.transforms
        for cluster_id in self.cluster_ids:
            features = self.features[torch.tensor(self.clusters[cluster_id])]
            for transform in transforms:
                features = transform(features)
            self.features[torch.tensor(self.clusters[cluster_id])] = features

    def global_transform(self, transforms=None):
        if transforms is None:
            transforms = self.transforms
        for transform in transforms:
            self.features = transform(self.features)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        cluster_id = self.labels[idx].item()
        if random.random() < self.positive_percent:
            cluster_id = self.labels[idx].item()
        else:
            cluster_id = self.cluster_ids[random.randint(0, len(self.cluster_ids)-1)]
        other_id = self.clusters[cluster_id][random.randint(0, len(self.clusters[cluster_id])-1)]
        feature = self.features[idx]
        o_feature = self.features[other_id]
        return feature, o_feature, self.labels[idx], self.labels[other_id]


class GraphDataset(ClusterDataset):
    def initialize(self):
        # assert self.knn+1 < self.features.shape[0]/2
        nbrs = NearestNeighbors(n_neighbors=self.knn+1, p=1, algorithm='kd_tree', n_jobs=-1).fit(self.features)
        distances, y_indices = nbrs.kneighbors(self.features)
        print('knn finished')
        y_indices = y_indices[:, 1:].flatten()
        x_indices = np.arange(len(y_indices))//self.knn
        indices = np.stack([x_indices, y_indices], axis=0)
        indices = np.concatenate([indices, indices[::-1,:]], axis=-1)
        indices = np.unique(indices, axis=-1)
        counts = np.bincount(indices.flatten()) // 2
        weights = np.ones((indices.shape[-1])) #
        if self.normalize:
            weights /= np.sqrt(counts[indices[0]] * counts[indices[1]])
        self.D = torch.tensor(counts)
        self.A = torch.sparse_coo_tensor(indices, weights, (len(self.labels), len(self.labels))).float().coalesce()
        if self.labels is None:
            self.C = None
        else:
            self.C = self.labels[indices[0]] != self.labels[indices[1]]

    def initialize_dense(self, to_dense=False):
        self.D = torch.ones((len(self.labels))) * (len(self.labels)-1)
        self.A = torch.ones((len(self.labels), len(self.labels)))
        if self.normalize:
            self.A /= (len(self.labels)-1)
        self.A = self.A.to_sparse()
        if self.labels is None:
            self.C = None
        else:
            self.C = self.labels[self.A.indices()[0]] != self.labels[self.A.indices()[1]]
        if to_dense:
            self.A = self.A.coalesce().to_dense()



    def __init__(self, dataframe, features, cluster_ids=None, knn=5, normalize=True, feature_norms=None, scales=None, discretize=False):
        super().__init__(dataframe, features, cluster_ids, feature_norms, scales)
        self.knn = knn
        self.normalize = normalize
        if discretize:
            unique_labels = np.unique(self.labels.numpy())
            reverse_map = {label:i for i,label in enumerate(unique_labels)}
            self.labels = torch.tensor([reverse_map[label] for label in self.labels.numpy()])
            self.count_labels = torch.max(self.labels)+1

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.labels is None:
            return self.A, self.features
        else:
            return self.A, self.features, self.C    
