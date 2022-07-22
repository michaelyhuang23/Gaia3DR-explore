import torch
import seaborn as sns
import os
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from collections import Counter
from pyvis.network import Network
from random import randint
import sklearn

from neural_dataset import ClusterDataset
from cluster_analysis import C_HDBSCAN,C_Spectral
from evaluation_metrics import ClusterEvalIoU, ClusterEvalMode, Purity
from utils import cart2spherical, UnionFind

# apply on m12f, scale of m12f is larger
# action scale is probably higher
# mass of host galaxy for normalization? Energy, action scales with mass of host galaxy

device = 'cpu'
sample_size = 1000
data_root = 'data'
dataset_name = 'm12i_cluster_data_large_mass_large_cluster_v2'
dataset_path = os.path.join(data_root, dataset_name)

#23, 2, 26, 4, 1, 13, 21, 7, 5, 12, 14, 19
#
#22, 15, 18, 20, 3, 8
easy_small_clusters = [13, 21, 7, 5]
easy_mid_clusters = [6, 17, 14, 11, 8, 9, 1, 2]
easy_large_clusters = [22, 15, 18, 20, 3, 8]

df = pd.read_hdf(dataset_path+'.h5', key='star')
df['rstar'] = np.linalg.norm([df['xstar'].to_numpy(),df['ystar'].to_numpy(),df['zstar'].to_numpy()],axis=0)
#df = df.loc[np.isin(df['cluster_id'], easy_large_clusters)].copy()
#df = df.loc[df['cluster_id']<20].copy()
print(Counter(df['cluster_id']))

df_std = pd.read_csv(dataset_path+'_std.csv')


feature_columns = ['estar', 'lzstar', 'lxstar', 'lystar', 'jzstar', 'jrstar', 'eccstar', 'rstar', 'feH', 'mgfe', 'xstar', 'ystar', 'zstar', 'vxstar', 'vystar', 'vzstar', 'vrstar', 'vphistar', 'vrstar', 'vthetastar']
sample_size = min(len(df), sample_size)
sample_ids = np.random.choice(len(df), min(len(df), sample_size), replace=False)
df_trim = df.iloc[sample_ids].copy()
df = df_trim
dataset = ClusterDataset(df_trim, feature_columns, 'cluster_id', feature_divs=df_std)
labels = dataset.labels
print(Counter(labels.numpy()))

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
	B = len(dataset)
	L_features = dataset.features[None,...].repeat(B,1,1).reshape(B*B, dataset.features.shape[-1])
	R_features = dataset.features[:,None,:].repeat(1,B,1).reshape(B*B, dataset.features.shape[-1])
	dist = model(L_features, R_features).reshape(B,B)
	return dist

colors = ['#%06X' % randint(0, 0xFFFFFF) for i in range(27)]


with torch.no_grad():
	model = torch.load(f'weights/model_contrastive_32_32_epoch{59}.pth') # 229
	model.eval()
	model.config(False)

	dist = compute_distance(model, dataset, sample_size)
#	dist[dist >= 0.7] = 1
	dist = np.minimum(dist, np.transpose(dist))

	#clusterer = C_HDBSCAN(metric='precomputed', min_cluster_size=2, min_samples=1, cluster_selection_method='leaf', cluster_selection_epsilon=0.01)
	clusterer = C_Spectral(n_components=26, assign_labels='kmeans')
	clusterer.add_data(1-dist)
	clusters = clusterer.fit()

	cluster_names = [f'cluster_{label}' for label in clusters]
	cluster_eval = ClusterEvalIoU(clusters, labels.numpy(), IoU_thres=0.3)
	print(f'avg precision:\n {cluster_eval.precision}, \n avg recall: \n{cluster_eval.recall}')
	print(f'TP: {cluster_eval.TP}, T: {cluster_eval.T}, P: {cluster_eval.P}')

	print(f'purity: {Purity(clusters, labels.numpy())()}')
	print(f'AMI: {sklearn.metrics.adjusted_mutual_info_score(labels.numpy(), clusters)}')
	print(f'RAND: {sklearn.metrics.adjusted_rand_score(labels.numpy(), clusters)}')

	cluster_eval2 = ClusterEvalMode(clusters, labels.numpy())
	print(f'avg precision:\n {cluster_eval2.precision}, \n avg recall: \n{cluster_eval2.recall}')
	print(f'TP: {cluster_eval2.TP}, T: {cluster_eval2.T}, P: {cluster_eval2.P}')


	# connectivity = UnionFind(dist.shape[0])
	# net = Network()
	# unique_labels = np.unique(dataset.labels)
	# for i,lab in enumerate(dataset.labels):
	# 	lab_cluster = int(clusters[i])
	# 	net.add_node(i, label=lab.item(), color=colors[lab.item()], title=str(lab.item()))
	# edges = []
	# for i in range(sample_size):
	# 	for j in range(i+1, sample_size):
	# 		if dist[i,j]<0.5:
	# 			net.add_edge(i, j, weight=1-dist[i, j], value=1-dist[i, j], title=str(np.round(1-dist[i, j],1)))
	# 		#edges.append((dist[i,j], i, j))
	# # edges.sort()
	# # for d, i, j in edges:
	# # 	if connectivity.connect(i,j) : continue
	# # 	connectivity.join(i,j)
	# # 	net.add_edge(i, j, weight=1-dist[i, j], value=1-dist[i, j], title=str(np.round(1-dist[i, j],1)))
	# net.toggle_physics(False)
	# net.show('cluster_graph.html')

# fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# clusters_names = [f'cluster {label}' for label in clusters]
# sns.scatterplot(data=df, ax=axes[0,0], x='lzstar', y='estar', hue=clusters_names)
# sns.scatterplot(data=df, ax=axes[0,1], x='lystar', y='estar', hue=clusters_names)
# sns.scatterplot(data=df, ax=axes[0,2], x='rstar', y='eccstar', hue=clusters_names)
# sns.scatterplot(data=df, ax=axes[1,0], x='jphistar', y='jzstar', hue=clusters_names)
# sns.scatterplot(data=df, ax=axes[1,1], x='jphistar', y='jrstar', hue=clusters_names)
# sns.scatterplot(data=df, ax=axes[1,2], x='vxstar', y='vphistar', hue=clusters_names)
# plt.show()
