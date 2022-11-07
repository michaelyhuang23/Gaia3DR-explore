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

def evaluate_param(n_cluster=50, n_projection_dim=3):
	gnn_kmeans_clusterer = C_GNN_KMeans(len(feature_columns), n_cluster, 3, 0.001, device)
	gnn_kmeans_clusterer.load_model('../weights/GNN_KMeans/', 87)
	evaluator = CaterpillarEvaluator(gnn_kmeans_clusterer, dataset, 10000, True)
	metric = evaluator.evaluate_all()
	return metric


metric = evaluate_param()

