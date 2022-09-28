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

dataset = PointDataset(feature_columns, 'cluster_id')

def evaluate_param(n_components):
    clusterer = C_GaussianMixture(n_components=n_components)
    evaluator = CaterpillarEvaluator(clusterer, dataset, 1000, True)
    evaluator.evaluate_all()

evaluate_param(70)
