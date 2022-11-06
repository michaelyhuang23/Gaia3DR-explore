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
	gnn_gmm_clusterer = C_GNN_GMM(len(feature_columns), n_cluster, n_projection_dim, 0.001, device)
	gnn_gmm_clusterer.load_model('../weights/GNN_GMM/', 5)
	evaluator = CaterpillarEvaluator(gnn_gmm_clusterer, dataset, 10000, True)
	metric = evaluator.evaluate_all()
	return metric


metric = evaluate_param()

