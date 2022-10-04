import sys
sys.path.append('..')

import json
import numpy as np
import pandas as pd
import torch

from tools.cluster_analysis import *
from tools.neural_dataset import *
from tools.evaluation_metrics import *
from tools.evaluate_caterpillar import *
from tools.train_caterpillar import *


feature_columns = ['estar', 'jrstar', 'jzstar', 'jphistar']

dataset = GraphDataset(feature_columns, cluster_ids='cluster_id', scales=None, knn=10, normalize=True, discretize=False)
snc_clusterer = C_SNC(len(feature_columns), 50, similar_weight=1, egnn_lr=0.01, clustergen_lr=0.01, clustergen_regularizer=0.00001, device='cpu')

trainer = CaterpillarTrainer(snc_clusterer, dataset, 1000000, val_size=4, k_fold=6)

trainer.train_set(trainer.val_set[0])

