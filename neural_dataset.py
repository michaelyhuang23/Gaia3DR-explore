from torch.utils.data import Dataset
import numpy as np
import torch

class ClusterDataset(Dataset):
	def __init__(self, dataframe, features, cluster_ids):
		if isinstance(features, np.ndarray):
			self.features = torch.tensor(features).float()
		else:
			self.features = torch.tensor(dataframe[features].to_numpy()).float()
		self.features -= torch.mean(self.features, dim=0)[None,...]
		self.features /= torch.std(self.features, dim=0)[None,...]
		if cluster_ids is None:
			self.labels = None
		elif isinstance(cluster_ids, str):
			self.labels = torch.tensor(dataframe[cluster_ids].to_numpy()).long()
		else:
			self.labels = torch.tensor(cluster_ids).long()
		if self.labels is not None:
			self.labels -= torch.min(self.labels)
		assert self.labels is None or len(self.labels) == self.features.shape[0]

	def __len__(self):
		return self.features.shape[0]

	def __getitem__(self, idx):
		if self.labels is None:
			return self.features[idx], None
		else:
			return self.features[idx], self.labels[idx]


