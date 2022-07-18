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

from neural_dataset import ClusterDataset
from neural_preprocess import ClassificationModel, ClusterMap, ClassificationHead, PairwiseHead, PairwiseModel, GaussianHead
from evaluation_metrics import ClassificationAcc, ClusterEvalIoU
from cluster_analysis import C_HDBSCAN, C_GaussianMixture

device = 'cpu'
EPOCH = 500
data_root = 'data'
dataset_name = 'm12i_cluster_data_large_mass_large_cluster_v2.h5'
dataset_path = os.path.join(data_root, dataset_name)

df = pd.read_hdf(dataset_path, key='star')
df['rstar'] = np.linalg.norm([df['xstar'].to_numpy(),df['ystar'].to_numpy(),df['zstar'].to_numpy()],axis=0)

print(df.columns)
feature_columns = ['estar', 'lzstar', 'lxstar', 'lystar', 'jzstar', 'jrstar', 'eccstar', 'rstar', 'feH', 'mgfe', 'xstar', 'ystar', 'zstar', 'vxstar', 'vystar', 'vzstar', 'vrstar', 'vphistar', 'vrstar', 'vthetastar']

cluster_ids = df['cluster_id'].to_numpy()
id_count = np.max(cluster_ids)
id_counter = Counter(cluster_ids)
print(id_counter)

weights = np.array([len(cluster_ids)/id_counter[c] for c in range(1,id_count+1)])
weights /= np.linalg.norm(weights)
weights = torch.tensor(weights).float()

dataset = ClusterDataset(df, feature_columns, 'cluster_id')
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

mapper = ClusterMap(len(feature_columns), [256, 256, 3], device=device)
#classifier = GaussianHead(mapper.output_size, id_count, weights=weights, device=device)
#model = ClassificationModel(len(feature_columns), id_count, device=device, mapper=mapper, classifier=classifier)
pairer = PairwiseHead(metric='euclidean')
model = PairwiseModel(len(feature_columns), device=device, mapper=mapper, pairloss=pairer)
clusterer = C_HDBSCAN(metric='euclidean', min_cluster_size=20, min_samples=10, cluster_selection_method='eom', cluster_selection_epsilon=0.01)
optimizer = Adam(model.parameters(), lr=0.000001, weight_decay=1e-5)

def train_epoch_step(epoch, dataloader, model, optimizer, device):
	model.train()
	model.config(True)
	dataloader_bar = tqdm(dataloader)
	t_loss = 0
	for features, labels in dataloader_bar:
		features.to(device)
		labels.to(device)
		loss = model(features, labels)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		dataloader_bar.set_description("Loss %s" % str(loss.item()))
		t_loss += loss.cpu().detach().item()
	print(f'epoch loss: {t_loss/len(dataloader)}')

def test_epoch_step_classification(epoch, dataloader, model, num_classes, device):
	model.eval()
	model.config(True)
	dataloader_bar = tqdm(dataloader)
	t_preds = []
	t_labels = []
	TP_class = torch.zeros((26)).long()
	for features, labels in dataloader_bar:
		features.to(device)
		labels.to(device)
		logits, probs, preds, scores = model(features)
		assert len(preds) == len(labels)
		TP_class[labels] += preds == labels
		class_acc = torch.sum(preds == labels) / len(preds)
		dataloader_bar.set_description("Acc %s" % str(class_acc.item()))
		t_preds.extend(list(preds.data))
		t_labels.extend(list(labels.data))
	metrics = ClassificationAcc(t_preds, t_labels, num_classes)
	sns.heatmap(metrics.count_matrix)
	plt.savefig(f'figures/gaussian_count_matrix_epoch_{epoch}')
	plt.clf()
	sns.heatmap(metrics.precision_matrix)
	plt.savefig(f'figures/gaussian_precision_matrix_epoch_{epoch}')
	plt.clf()
	sns.heatmap(metrics.recall_matrix)
	plt.savefig(f'figures/gaussian_recall_matrix_epoch_{epoch}')
	plt.clf()
	print(f'count:\n {pd.DataFrame(metrics.count_matrix)}, \n precision:\n {pd.DataFrame(np.round(metrics.precision_matrix,2))}, \n recall: \n{pd.DataFrame(np.round(metrics.recall_matrix,2))}')
	torch.save(model, f'weights/model_gaussian_256_256_3_epoch{epoch}.pth')

def test_epoch_step_cluster(epoch, dataset, model, num_classes, device, sample_size=10000):
	model.eval()
	model.config(False)
	sample_ids = np.random.choice(len(dataset), min(len(dataset), sample_size), replace=False)
	features = dataset.features[sample_ids]
	labels = dataset.labels[sample_ids]
	features.to(device)
	labels.to(device)
	mapped_features = model(features)
	print(mapped_features.shape)
	clusterer.add_data(mapped_features.numpy())
	preds = clusterer.fit()
	cluster_eval = ClusterEvalIoU(preds, labels.numpy())
	print(f'avg precision:\n {cluster_eval.precision}, \n avg recall: \n{cluster_eval.recall}')
	print(f'TP: {cluster_eval.TP}, T: {cluster_eval.T}, P: {cluster_eval.P}')
	torch.save(model, f'weights/model_pairwise_256_256_3_epoch{epoch}.pth')
	

for epoch in range(EPOCH):
	train_epoch_step(epoch, dataloader, model, optimizer, device)
	if (epoch+1) % 1 == 0:
		with torch.no_grad():
			test_epoch_step_cluster(epoch, dataset, model, id_count, device) # we should use test data loader







