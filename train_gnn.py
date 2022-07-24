import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD, Adam
from collections import Counter

from neural_dataset import *
from gnn_cluster import *
from evaluation_metrics import ClassificationAcc, ClusterEvalIoU
from cluster_analysis import C_HDBSCAN, C_GaussianMixture


if __name__ == '__main__':
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	EPOCH = 500
	data_root = 'data/simulation'
	dataset_name = 'm12f_cluster_data_large_cluster_v2'
	dataset_path = os.path.join(data_root, dataset_name)

	test_dataset_name = 'm12i_cluster_data_large_cluster_v2'
	test_dataset_path = os.path.join(data_root, test_dataset_name)

	print(f'running with {device}')

	df = pd.read_hdf(dataset_path+'.h5', key='star')
	df_std = pd.read_csv(dataset_path+'_std.csv')
	df_test = pd.read_hdf(test_dataset_path+'.h5', key='star')
	df_test_std = pd.read_csv(test_dataset_path+'_std.csv')

	print(df.columns)
	feature_columns = ['estar', 'lzstar', 'lxstar', 'lystar', 'jzstar', 'jrstar', 'eccstar', 'rstar', 'feH', 'mgfe', 'xstar', 'ystar', 'zstar', 'vxstar', 'vystar', 'vzstar', 'vrstar', 'vphistar', 'vthetastar', 'omegaphistar', 'omegarstar', 'omegazstar', 'thetaphistar', 'thetarstar', 'thetazstar', 'zmaxstar']

	cluster_ids = df['cluster_id'].to_numpy()
	id_count = np.max(cluster_ids)
	id_counter = Counter(cluster_ids)
	print(id_counter)

	weights = np.array([len(cluster_ids)/id_counter[c] for c in range(1,id_count+1)])
	weights /= np.linalg.norm(weights)
	weights = torch.tensor(weights).float()

	dataset = GraphDataset(df, feature_columns, 'cluster_id', 5, feature_divs=df_std)
	test_dataset = GraphDataset(df_test, feature_columns, 'cluster_id', 5, feature_divs=df_test_std)


	model = GCNEdge(len(feature_columns), graph_layer_sizes=[32,32], linear_layer_sizes=[32], similar_weight=1, device=device)
	optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

	def train_epoch_step(epoch, dataset, model, optimizer, device):
		model.train()
		model.config(True)
		A, X, C = dataset[0]
		model.add_graph(A.to(device))
		model.add_connectivity(C.to(device))
		loss = model(X.to(device))
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		print(f'epoch loss: {loss}')

	for epoch in range(EPOCH):
		train_epoch_step(epoch, dataset, model, optimizer, device)
		
		# with torch.no_grad():
		# 	dataloader.dataset.cluster_transform(transforms=[GlobalJitterTransform(), GlobalScaleTransform()])
		# 	dataloader.dataset.global_transform(transforms=[JitterTransform()])
		# 	if (epoch+1) % 2 == 0:
		# 		print('training set acc:')
		# 		test_epoch_step_contrastive(epoch, dataloader, model, device)
		# 		print('testing set acc:')
		# 		test_epoch_step_contrastive(epoch, test_dataloader, model, device)

		# 		dataset = ContrastDataset(df, feature_columns, 'cluster_id', feature_divs=df_std, positive_percent=0)
		# 		dataloader = DataLoader(dataset, batch_size=128, shuffle=True)









