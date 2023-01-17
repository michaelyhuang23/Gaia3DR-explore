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


feature_columns = ['estar', 'jrstar', 'jzstar', 'jphistar', 'rstar', 'vstar', 'vxstar', 'vystar', 'vzstar', 'vrstar', 'vphistar', 'phistar', 'zstar']

def filterer(df):
    return df.loc[df['redshiftstar']<2].copy()

dataset = PointDataset(feature_columns, 'cluster_id')



def evaluate_param(n_components):
    clusterer = C_GaussianMixture(n_components=n_components)
    evaluator = CaterpillarEvaluator(clusterer, dataset, 1000000, filterer=filterer, run_on_test=True)
    f_metrics = evaluator.evaluate_all()
    return f_metrics

results = {}
for n_components in ([2,4,8,16] + list(range(30, 500, 10))):
    metric = evaluate_param(n_components)
    results[n_components] = metric


with open('../results/gmm_caterpillar.json', 'w') as f:
    json.dump(results, f)