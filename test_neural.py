import torch
import seaborn as sns
import os
import pandas as pd
import numpy as np
import plotly.express as px

from neural_dataset import ClusterDataset
from cluster_analysis import C_HDBSCAN, C_GaussianMixture
from evaluation_metrics import ClusterEvalIoU
from utils import cart2spherical


device = 'cpu'
sample_size = 10000
EPOCH = 500
data_root = 'data'
dataset_name = 'm12i_cluster_data_large_mass_large_cluster_v2.h5'
dataset_path = os.path.join(data_root, dataset_name)

df = pd.read_hdf(dataset_path, key='star')
df['rstar'] = np.linalg.norm([df['xstar'].to_numpy(),df['ystar'].to_numpy(),df['zstar'].to_numpy()],axis=0)
feature_columns = ['estar', 'lzstar', 'lxstar', 'lystar', 'jzstar', 'jrstar', 'eccstar', 'rstar', 'feH', 'mgfe', 'xstar', 'ystar', 'zstar', 'vxstar', 'vystar', 'vzstar', 'vrstar', 'vphistar', 'vrstar', 'vthetastar']

dataset = ClusterDataset(df, feature_columns, 'cluster_id')
sample_ids = np.random.choice(len(dataset), min(len(dataset),sample_size))

with torch.no_grad():
	model = torch.load(f'weights/model_gaussian_256_256_3_epoch{79}.pth')
	model.eval()
	model.config(True)

	X, probs, preds, scores = model(dataset.features[sample_ids])
	mapped_features = model.mapper(dataset.features[sample_ids])

	labels = dataset.labels[sample_ids]

	label_names = [f'cluster_{label}' for label in preds]
	label_range = [f'cluster_{label}' for label in range(model.classifier.W.shape[0])]
	fig = px.scatter_3d(x=model.classifier.W[:,0], y=model.classifier.W[:,1], z=model.classifier.W[:,2], color=label_range)
	fig.show()
	fig = px.scatter_3d(x=mapped_features[:,0], y=mapped_features[:,1], z=mapped_features[:,2], color=label_names)
	fig.show()

	#clusterer = C_HDBSCAN(metric='euclidean', min_cluster_size=20, min_samples=10, cluster_selection_method='leaf', cluster_selection_epsilon=0.01)
	clusterer = C_GaussianMixture(n_components=26)
	clusterer.add_data(mapped_features.numpy())
	clusters = clusterer.fit()
	cluster_names = [f'cluster_{label}' for label in clusters]
	cluster_eval = ClusterEvalIoU(clusters, labels.numpy())
	print(f'avg precision:\n {cluster_eval.precision}, \n avg recall: \n{cluster_eval.recall}')
	print(f'TP: {cluster_eval.TP}, T: {cluster_eval.T}, P: {cluster_eval.P}')
	fig = px.scatter_3d(x=mapped_features[:,0], y=mapped_features[:,1], z=mapped_features[:,2], color=cluster_names)
	fig.show()


