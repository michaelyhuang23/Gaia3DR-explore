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
	sample_size = 100000
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	EPOCH = 1000
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

	sample_ids = np.random.choice(len(df), min(len(df), sample_size), replace=False)
	df = df.iloc[sample_ids].copy()

	print(df.columns)
	feature_columns = ['estar', 'lzstar', 'lxstar', 'lystar', 'jzstar', 'jrstar', 'eccstar', 'rstar', 'feH', 'mgfe', 'xstar', 'ystar', 'zstar', 'vxstar', 'vystar', 'vzstar', 'vrstar', 'vphistar', 'vthetastar', 'omegaphistar', 'omegarstar', 'omegazstar', 'thetaphistar', 'thetarstar', 'thetazstar', 'zmaxstar']

	cluster_ids = df['cluster_id'].to_numpy()
	id_count = np.max(cluster_ids)
	id_counter = Counter(cluster_ids)
	print(id_counter)


	dataset = GraphDataset(df, feature_columns, 'cluster_id', 1, feature_divs=df_std)
	#test_dataset = GraphDataset(df_test, feature_columns, 'cluster_id', 1, feature_divs=df_test_std)


	model = GCNEdge(len(feature_columns), graph_layer_sizes=[32,32], linear_layer_sizes=[32], similar_weight=1, device=device)
	optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

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
		return loss.item()


	for epoch in range(EPOCH):
		loss = train_epoch_step(epoch, dataset, model, optimizer, device)
		if (epoch) % 10 == 0:
			print(f'epoch {epoch}, loss: {loss}')
		if (epoch) % 100 == 0:		
			with torch.no_grad():
				model.config(False)
				model.eval()
				A, X, C = test_dataset[0]
				model.add_graph(A.to(device))
				model.add_connectivity(C.to(device))
				SX = model(X.to(device))
				preds = np.rint(SX.numpy()).astype(np.int32)
				metrics = ClassificationAcc(preds, C.numpy().astype(np.int32), 2)
				print(f' precision: {metrics.precision}\n recall: {metrics.recall}')









