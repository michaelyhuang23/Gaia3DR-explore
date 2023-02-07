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
print(f'using device {device}')

feature_columns = ['estar', 'jrstar', 'jzstar', 'jphistar', 'rstar', 'vstar', 'vxstar', 'vystar', 'vzstar', 'vrstar', 'vphistar', 'phistar', 'zstar']


def filterer(df):
    return df.loc[df['redshiftstar']<2].copy()

dataset = GraphDataset(feature_columns, cluster_ids='cluster_id', scales=None, knn=None, randomn=None, normalize=True, discretize=False)


def evaluate_param(clustergen_regularizer=0.00001):
	snc_clusterer = C_SNC(len(feature_columns), 50, similar_weight=1, egnn_lr=0.001, egnn_regularizer=0, clustergen_lr=0.003, clustergen_regularizer=clustergen_regularizer, device=device)
	snc_clusterer.load_model('../weights/SNC/', 99)
	evaluator = CaterpillarEvaluator(snc_clusterer, dataset, 1000000, filterer=filterer, run_on_test=True)
	metric = evaluator.evaluate_all()
	return metric


metric = evaluate_param()
print(metric)

results = {0.00001 : metric}

with open('../results/snc_caterpillar.json', 'w') as f:
    json.dump(results, f)






