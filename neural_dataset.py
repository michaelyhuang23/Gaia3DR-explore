from torch.utils.data import Dataset
import numpy as np
import torch

class ClusterDataset(Dataset):
	def __init__(self, dataframe, features, cluster_ids, feature_divs=None):
		super().__init__()
		self.standard_feature_divs = {'estar':1e5, 'lzstar':2000, 'lxstar':2000, 'lystar':2000, 'jzstar':2000, 'jrstar':2000, 'eccstar':1, 'rstar':4, 'feH':1, 'mgfe':0.5, 'xstar':10, 'ystar':10, 'zstar':10, 'vxstar':200, 'vystar':200, 'vzstar':200, 'vrstar':200, 'vphistar':200, 'vrstar':200, 'vthetastar':200}
		if feature_divs is None:
			self.feature_divs = torch.tensor([self.standard_feature_divs[feature] for feature in features])
		else:
			self.feature_divs = torch.tensor(feature_divs)
		if isinstance(features, np.ndarray):
			self.features = torch.tensor(features).float()
		else:
			self.features = torch.tensor(dataframe[features].to_numpy()).float()
		self.features /= self.feature_divs[None,...]
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

class ContrastDataset(Dataset):
	def __init__(self, dataframe, features, cluster_ids, feature_divs=None):
		super().__init__()
		self.standard_feature_divs = {'estar':1e5, 'lzstar':2000, 'lxstar':2000, 'lystar':2000, 'jzstar':2000, 'jrstar':2000, 'eccstar':1, 'rstar':4, 'feH':1, 'mgfe':0.5, 'xstar':10, 'ystar':10, 'zstar':10, 'vxstar':200, 'vystar':200, 'vzstar':200, 'vrstar':200, 'vphistar':200, 'vrstar':200, 'vthetastar':200}
		if feature_divs is None:
			self.feature_divs = torch.tensor([self.standard_feature_divs[feature] for feature in features])
		else:
			self.feature_divs = torch.tensor(feature_divs)
		if isinstance(features, np.ndarray):
			self.features = torch.tensor(features).float()
		else:
			self.features = torch.tensor(dataframe[features].to_numpy()).float()
		self.features /= self.feature_divs[None,...]
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

	def __len__(self):
		return self.features.shape[0]

	def __getitem__(self, idx):
		cluster_id = np.random.choice(self.cluster_ids, 1)[0]
		other_id = np.random.choice(self.clusters[cluster_id], 1)[0]
		assert self.labels[other_id].item() == cluster_id
		return self.features[idx], self.features[other_id], self.labels[idx], self.labels[other_id]

