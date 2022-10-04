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


feature_columns = ['estar', 'jrstar', 'jzstar', 'jphistar']

dataset = PointDataset(feature_columns, 'cluster_id')

def evaluate_param(n_components):
    clusterer = C_GaussianMixture(n_components=n_components)
    evaluator = CaterpillarEvaluator(clusterer, dataset, 1000000, True)
    f_metrics = evaluator.evaluate_all()
    return f_metrics

results = {}
for n_components in [2,4,8,16,30,40,50,60,70,80,90,100,110,120,130,140,160,180,200]:
    metric = evaluate_param(n_components)
    results[n_components] = metric


with open('../results/gmm_caterpillar.json', 'w') as f:
    json.dump(results, f)