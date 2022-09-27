import numpy as np
import os, json
import torch
import pandas as pd
from cluster_analysis import *


class CaterpillarEvaluator:
    def __init__(self, clusterer, dataset):
        # clusterer is a trained Clusterer, dataset is an empty Dataset
        super().__init__()
        self.test_ids = [1079897, 1232423, 1599902, 196078]
        self.dataset_root = '../data/caterpillar/labeled_caterpillar_data'
        self.clusterer = clusterer
        self.dataset = dataset

    def evaluate(self, dataset_id):
        dataset_name = f'{dataset_id}_all'
        df = pd.read_hdf(os.path.join(self.dataset_root, dataset_name+'.h5'), key='star')
        with open(os.path.join(self.dataset_root, dataset_name+'_norm.json'), 'r') as f:
            df_norm = json.load(f)
        self.dataset.load_data(df, df_norm)
        self.clusterer.add_data(self.dataset)
        return self.clusterer.fit()
        
    def evaluate_all(self):
        metrics = []
        for test_id in self.test_ids:
            metric = self.evaluate(clusterer, test_id)
            metrics.append(metric)
        f_metric = metrics.aggregate()
        return f_metric


