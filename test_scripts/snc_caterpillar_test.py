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


device = 'cuda' if torch.cuda.is_available() else 'cpu'

feature_columns = ['estar', 'jrstar', 'jzstar', 'jphistar']

dataset = GraphDataset(feature_columns, cluster_ids='cluster_id', scales=None, knn=100, normalize=True, discretize=False)


def evaluate_param(clustergen_regularizer=0.0001):
	snc_clusterer = C_SNC(len(feature_columns), 50, similar_weight=1, egnn_lr=0.003, egnn_regularizer=0.1, clustergen_lr=0.01, clustergen_regularizer=clustergen_regularizer, device=device)
	snc_clusterer.load_model('../weights/SNC/', 2)
	evaluator = CaterpillarEvaluator(snc_clusterer, dataset, 1000000, True)
	metric = evaluator.evaluate_all()
	return metric


metric = evaluate_param()
print(metric)







