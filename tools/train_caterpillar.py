import random
import numpy as np
import os, json
import torch
import pandas as pd
from tools.cluster_analysis import *
from tools.evaluation_metrics import *
from tools.neural_dataset import *

class CaterpillarTrainer:
    def __init__(self, clusterer_model, dataset, sample_size, val_size = 4, k_fold = 6, writer = None):
        # clusterer is an untrained Clusterer, dataset is an empty Dataset
        super().__init__()
        self.train_ids = [1104787, 1130025, 1195075, 1195448, 1232164, 1268839, 1292085,\
                    1354437, 1387186, 1422331, 1422429, 1599988, 1631506, 1631582, 1725139,\
                    1725272, 196589, 264569, 388476, 447649, 5320, 581141, 581180, 65777, 795802,\
                    796175, 94638, 95289]
        random.shuffle(self.train_ids)
        self.dataset_root = '../data/caterpillar/labeled_caterpillar_data'
        self.clusterer = clusterer_model
        self.dataset = dataset
        self.sample_size = sample_size
        self.val_size = val_size
        self.k_fold = k_fold
        self.val_set = []
        self.writer = writer
        self.counter = 0
        for fold in range(k_fold):
            start = len(self.train_ids)//k_fold*fold
            end = start + self.val_size
            self.val_set.append(self.train_ids[start:end])

    def train_step(self, dataset_id, train_epoch=1, repetition=5):
        print(f'training {dataset_id}')
        dataset_name = f'labeled_{dataset_id}_all'
        df_ = pd.read_hdf(os.path.join(self.dataset_root, dataset_name+'_0.h5'), key='star')
        with open(os.path.join(self.dataset_root, dataset_name+'_0_norm.json'), 'r') as f:
            df_norm = json.load(f)
        for i in range(train_epoch):
            df = sample_space(df_, radius=0.005, radius_sun=0.0082, zsun_range=0.016/1000, sample_size=self.sample_size, filter_size=10)
            print(len(df), len(df_))
            self.dataset.load_data(df, df_norm)
            self.clusterer.add_data(self.dataset)
            for j in range(repetition):
                loss = self.clusterer.train()
                self.writer.add_scalar('Loss', loss, self.counter)
                self.counter += 1
                print(f'run {i}, instance {j}, loss: {loss}')

    def train_set(self, val_ids):
        f_metric = self.evaluate_all(val_ids)     
        for train_id in self.train_ids:
            if train_id in val_ids: continue
            self.train_step(train_id, train_epoch=1, repetition=10)
        return f_metric

    def evaluate(self, dataset_id, eval_epoch=1):
        print(f'evaluating {dataset_id}')
        dataset_name = f'labeled_{dataset_id}_all'
        df_ = pd.read_hdf(os.path.join(self.dataset_root, dataset_name+'.h5'), key='star')
        with open(os.path.join(self.dataset_root, dataset_name+'_norm.json'), 'r') as f:
            df_norm = json.load(f)
        metrics = []
        for i in range(eval_epoch):
            print(f'cluster iteration {i}')
            df = sample_space(df_, radius=0.005, radius_sun=0.0082, zsun_range=0.016/1000, sample_size=self.sample_size, filter_size=10)
            self.dataset.load_data(df, df_norm)
            self.clusterer.add_data(self.dataset)
            labels, loss = self.clusterer.fit()
            self.writer.add_scalar('Val/Loss', loss, self.counter)
            self.counter += 1
            if torch.is_tensor(labels):
                labels = labels.detach().cpu().numpy()
            t_labels = self.dataset.labels
            if torch.is_tensor(t_labels):
                t_labels = t_labels.detach().cpu().numpy()            
            cluster_eval = ClusterEvalAll(labels, self.dataset.labels.cpu().numpy())
            metric = cluster_eval()
            metrics.append(metric)           
        return ClusterEvalAll.aggregate(metrics)
        
    def evaluate_all(self, val_ids):
        metrics = []
        for val_id in val_ids:
            metric = self.evaluate(val_id)
            print(metric)
            metrics.append(metric)
        f_metric = ClusterEvalAll.aggregate(metrics)
        return f_metric


