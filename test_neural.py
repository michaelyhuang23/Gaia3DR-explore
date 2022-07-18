import torch
import seaborn as sns
import os
import pandas as pd
import numpy as np
import plotly.express as px

from neural_dataset import ClusterDataset
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
	model = torch.load(f'weights/model_256_256_3_epoch{499}.pth')
	model.eval()
	model.config(False)

	mapped_features = model(dataset.features[sample_ids]).detach().numpy()
	r, phi, theta = cart2spherical(mapped_features[:,0], mapped_features[:,1], mapped_features[:,2])
	fig = px.scatter_3d(x=r, y=phi, z=theta)
	fig.show()


