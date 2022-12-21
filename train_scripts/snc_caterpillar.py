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

print(f'using device: {device}')

feature_columns = ['estar', 'jrstar', 'jzstar', 'jphistar', 'rstar', 'vstar', 'vxstar', 'vystar', 'vzstar', 'vrstar', 'vphistar', 'phistar', 'zstar']

def filterer(df):
    return df.loc[df['redshiftstar']<2].copy()

dataset = GraphDataset(feature_columns, cluster_ids='cluster_id', scales=None, knn=None, randomn=None, normalize=True, discretize=False)
snc_clusterer = C_SNC(len(feature_columns), 150, similar_weight=1, egnn_lr=0.001, egnn_regularizer=0, clustergen_lr=0.001, clustergen_regularizer=0.00001, device=device)

trainer = CaterpillarTrainer(snc_clusterer, dataset, 10000000, val_size=4, k_fold=6, filterer=filterer, writer=writer)

EPOCH = 100
for epoch in range(EPOCH):
	print(f'EPOCH {epoch+1}')
	metric = trainer.train_set(trainer.val_set[0])
	trainer.clusterer.save_model('../weights/SNC/', epoch)
	print(metric)
	writer.add_scalar('Acc/IoU_recall', metric['IoU_recall'], epoch)
	writer.add_scalar('Acc/IoU_precision', metric['IoU_precision'], epoch)

# the best epoch is 2 (0 based)

