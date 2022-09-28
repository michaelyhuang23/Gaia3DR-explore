import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import torch

from tools.cluster_analysis import *
from tools.neural_dataset import *
from tools.evaluation_metrics import *
from tools.evaluate_caterpillar import *


feature_columns = ['estar', 'jrstar', 'jzstar', 'jphistar']

dataset = PointDataset(feature_columns, 'cluster_id',)

def evaluate_param(min_cluster_size, min_samples):
    clusterer = C_HDBSCAN(metric='euclidean', min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_method='eom')
    evaluator = CaterpillarEvaluator(clusterer, dataset, 1000, True)
    f_metrics = evaluator.evaluate_all()
    print(f_metrics)

evaluate_param(3, None)
