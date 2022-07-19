import torch
import seaborn as sns
import os
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from collections import Counter
from pyvis.network import Network

from neural_dataset import ClusterDataset
from cluster_analysis import C_HDBSCAN
from evaluation_metrics import ClusterEvalIoU
from utils import cart2spherical, UnionFind



device = 'cpu'
sample_size = 100
data_root = 'data'
dataset_name = 'm12i_cluster_data_large_mass_large_cluster_v2.h5'
dataset_path = os.path.join(data_root, dataset_name)

df = pd.read_hdf(dataset_path, key='star')
df['rstar'] = np.linalg.norm([df['xstar'].to_numpy(),df['ystar'].to_numpy(),df['zstar'].to_numpy()],axis=0)
df = df.loc[np.isin(df['cluster_id'], [1,2,3,5])].copy()
print(Counter(df['cluster_id']))


feature_columns = ['estar', 'lzstar', 'lxstar', 'lystar', 'jzstar', 'jrstar', 'eccstar', 'rstar', 'feH', 'mgfe', 'xstar', 'ystar', 'zstar', 'vxstar', 'vystar', 'vzstar', 'vrstar', 'vphistar', 'vrstar', 'vthetastar']

sample_ids = np.random.choice(len(df), min(len(df), sample_size), replace=False)
df_trim = df.iloc[sample_ids].copy()
df = df_trim
dataset = ClusterDataset(df_trim, feature_columns, 'cluster_id')
labels = dataset.labels

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
df['lstar'] = np.linalg.norm([df['lzstar'].to_numpy(),df['lystar'].to_numpy(),df['lxstar'].to_numpy()],axis=0)
df['cluster_id_name'] = np.array([f'cluster {id}' for id in df['cluster_id'].to_numpy()])
df['c_lzstar'] = df['lzstar'].to_numpy()*np.abs(df['estar'].to_numpy())**2.3
df['c_lystar'] = df['lystar'].to_numpy()*np.abs(df['estar'].to_numpy())**2.3
df['c_lxstar'] = df['lxstar'].to_numpy()*np.abs(df['estar'].to_numpy())**2.3
df['s_jzrstar'] = df['jzstar'].to_numpy() - df['jrstar'].to_numpy() # mostly spherical because you can try squaring it
df['a_jzrstar'] = df['jzstar'].to_numpy() + df['jrstar'].to_numpy() # mostly spherical because you can try squaring it
df['rstar'] = np.linalg.norm([df['xstar'].to_numpy(),df['ystar'].to_numpy(),df['zstar'].to_numpy()],axis=0)
sns.scatterplot(data=df, ax=axes[0,0], x='lzstar', y='estar', hue='cluster_id_name')
sns.scatterplot(data=df, ax=axes[0,1], x='lystar', y='estar', hue='cluster_id_name')
sns.scatterplot(data=df, ax=axes[0,2], x='rstar', y='eccstar', hue='cluster_id_name')
sns.scatterplot(data=df, ax=axes[1,0], x='jphistar', y='jzstar', hue='cluster_id_name')
sns.scatterplot(data=df, ax=axes[1,1], x='jphistar', y='jrstar', hue='cluster_id_name')
sns.scatterplot(data=df, ax=axes[1,2], x='vxstar', y='vphistar', hue='cluster_id_name')
plt.show()

def compute_distance(model, dataset, sample_size):
	dist = np.zeros((sample_size, sample_size))
	for i in range(sample_size):
		for j in range(sample_size):
			dist[i, j] = model(dataset.features[i:i+1], dataset.features[j:j+1])
	return dist

colors = ['#00ff1e', '#162347', '#dd4b39', '#afffff', '#cfffff']

with torch.no_grad():
	model = torch.load(f'weights/model_contrastive_64_64_64_epoch{499}.pth')
	model.eval()
	model.config(False)

	dist = compute_distance(model, dataset, sample_size)

	clusterer = C_HDBSCAN(metric='precomputed', min_cluster_size=2, min_samples=1, cluster_selection_method='leaf', cluster_selection_epsilon=0.01)
	clusterer.add_data(dist)
	clusters = clusterer.fit()

	cluster_names = [f'cluster_{label}' for label in clusters]
	cluster_eval = ClusterEvalIoU(clusters, labels.numpy())
	print(f'avg precision:\n {cluster_eval.precision}, \n avg recall: \n{cluster_eval.recall}')
	print(f'TP: {cluster_eval.TP}, T: {cluster_eval.T}, P: {cluster_eval.P}')

	connectivity = UnionFind(dist.shape[0])
	net = Network()
	for i,lab in enumerate(dataset.labels):
		net.add_node(i, label=lab.item(), color=colors[lab.item()], title=str(lab.item()+1))
	edges = []
	for i in range(sample_size):
		for j in range(i+1, sample_size):
			edges.append((dist[i,j]+dist[j,i], i, j))
	edges.sort()
	for d, i, j in edges:
		if connectivity.connect(i,j) : continue
		connectivity.join(i,j)
		net.add_edge(i, j, weight=2-dist[i, j]-dist[j, i], value=2-dist[i, j]-dist[j, i], title=str(np.round(2-dist[i, j]-dist[j, i],1)))
	net.toggle_physics(False)
	net.show('cluster_graph.html')


