import sys
sys.path.append('..')

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from tools.cluster_analysis import *
from tools.neural_dataset import *
from tools.evaluation_metrics import *
from tools.evaluate_caterpillar import *
from tools.train_caterpillar import *


writer = SummaryWriter()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

feature_columns = ['estar', 'jrstar', 'jzstar', 'jphistar']

dataset = GraphDataset(feature_columns, cluster_ids='cluster_id', scales=None, knn=100, randomn=None, normalize=True, discretize=False)
gnn_gmm_clusterer = C_GNN_KMeans(len(feature_columns), 50, 3, 0.0001, device)

trainer = CaterpillarTrainer(gnn_gmm_clusterer, dataset, 10000, val_size=4, k_fold=6, writer=writer)

EPOCH = 100
for epoch in range(EPOCH):
	print(f'EPOCH {epoch+1}')
	metric = trainer.train_set(trainer.val_set[0])
	trainer.clusterer.save_model('../weights/GNN_KMeans/', epoch)
	print(metric)
	writer.add_scalar('Acc/IoU_recall', metric['IoU_recall'], epoch)
	writer.add_scalar('Acc/IoU_precision', metric['IoU_precision'], epoch)

# the best epoch is 2 (0 based)

