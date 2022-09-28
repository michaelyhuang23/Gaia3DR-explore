import numpy as np
import os, json
import torch
import pandas as pd
from tools.cluster_analysis import *
from tools.evaluation_metrics import *


class CaterpillarTrainer:
    def __init__(self, clusterer, dataset, sample_size, val_size = 1):
        # clusterer is a trained Clusterer, dataset is an empty Dataset
        super().__init__()
        self.train_ids = [1104787, 1130025, 1195075, 1195448, 1232164, 1268839, 1292085,\
                    1354437, 1387186, 1422331, 1422429, 1599988, 1631506, 1631582, 1725139,\
                    1725272, 196589, 264569, 388476, 447649, 5320, 581141, 581180, 65777, 795802,\
                    796175, 94638, 95289]
        self.dataset_root = '../data/caterpillar/labeled_caterpillar_data'
        self.clusterer = clusterer
        self.dataset = dataset
        self.sample_size = sample_size

    def evaluate(self, dataset_id, eval_epoch=10):
        print(f'evaluating {dataset_id}')
        dataset_name = f'labeled_{dataset_id}_all'
        df_ = pd.read_hdf(os.path.join(self.dataset_root, dataset_name+'.h5'), key='star')
        with open(os.path.join(self.dataset_root, dataset_name+'_norm.json'), 'r') as f:
            df_norm = json.load(f)
        metrics = []
        for i in range(eval_epoch):
            print(f'cluster iteration {i}')
            df = sample_space(df_, radius=0.005, radius_sun=0.0082, sample_size=self.sample_size)
            self.dataset.load_data(df, df_norm)
            self.clusterer.add_data(self.dataset)
            labels = self.clusterer.fit()
            cluster_eval = ClusterEvalAll(labels, self.dataset.labels)
            metrics.append(cluster_eval())           
        return ClusterEvalAll.aggregate(metrics)
        
    def evaluate_all(self):
        metrics = []
        if self.run_on_test:
            for test_id in self.test_ids:
                metric = self.evaluate(test_id)
                print(metric)
                metrics.append(metric)
        else:
            for train_id in self.train_ids:
                metric = self.evaluate(train_id)
                print(metric)
                metrics.append(metric)
        f_metric = ClusterEvalAll.aggregate(metrics)
        return f_metric


