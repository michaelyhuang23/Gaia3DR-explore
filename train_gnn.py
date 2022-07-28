import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD, Adam
from collections import Counter

from neural_dataset import *
from gnn_cluster import *
from evaluation_metrics import ClassificationAcc, ClusterEvalIoU
from cluster_analysis import C_HDBSCAN, C_GaussianMixture


if __name__ == '__main__':
    np.random.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCH = 4000
    data_root = 'data/simulation'
    dataset_name = 'm12f_cluster_data_large_cluster_v2'
    dataset_path = os.path.join(data_root, dataset_name)

    test_dataset_name = 'm12i_cluster_data_large_cluster_v2'
    test_dataset_path = os.path.join(data_root, test_dataset_name)

    print(f'running with {device}')

    df_ = pd.read_hdf(dataset_path+'.h5', key='star')
    df_std = pd.read_csv(dataset_path+'_std.csv')
    df_test = pd.read_hdf(test_dataset_path+'.h5', key='star')
    df_test_std = pd.read_csv(test_dataset_path+'_std.csv')

    sample_size = 1000
    sample_size = min(sample_size, len(df_))
    sample_ids = np.random.choice(len(df_), min(len(df_), sample_size), replace=False)
    df = df_.iloc[sample_ids].copy()

    sample_size = 1000
    sample_size = min(sample_size, len(df_test))
    sample_ids = np.random.choice(len(df_test), min(len(df_test), sample_size), replace=False)
    df_test = df_test.iloc[sample_ids].copy()

    print(df.columns)
    feature_columns = ['estar', 'lzstar', 'lxstar', 'lystar', 'jzstar', 'jrstar', 'eccstar', 'rstar', 'feH', 'mgfe', 'xstar', 'ystar', 'zstar', 'vxstar', 'vystar', 'vzstar', 'vrstar', 'vphistar', 'vthetastar', 'omegaphistar', 'omegarstar', 'omegazstar', 'thetaphistar', 'thetarstar', 'thetazstar', 'zmaxstar']

    dataset = GraphDataset(df, feature_columns, 'cluster_id', 50, feature_divs=df_std)
    dataset.initialize()
    test_dataset = GraphDataset(df_test, feature_columns, 'cluster_id', 50, feature_divs=df_test_std)
    test_dataset.initialize()


    model = GCNEdgeDot(len(feature_columns), graph_layer_sizes=[64,64,64], similar_weight=1, device=device)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    def train_epoch_step(epoch, dataset, model, optimizer, device):
        model.train()
        model.config(True)
        A, X, C = dataset[0]
        model.add_graph(A.to(device))
        model.add_connectivity(C.to(device))
        loss = model(X.to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()


    for epoch in range(EPOCH):
        if (epoch) % 50 == 0:      
            with torch.no_grad():
                model.config(False)
                model.eval()
                A, X, C = dataset[0]
                model.add_graph(A.to(device))
                model.add_connectivity(C.to(device))
                SX = model(X.to(device))
                preds = np.rint(SX.numpy()).astype(np.int32)
                metrics = ClassificationAcc(preds, C.numpy().astype(np.int32), 2)
                print(f'train acc: {metrics.precision}\n{metrics.count_matrix}')
                # print(np.mean(SX.numpy()), np.std(SX.numpy()))

                model.config(False)
                model.eval()
                A, X, C = test_dataset[0]
                model.add_graph(A.to(device))
                model.add_connectivity(C.to(device))
                SX = model(X.to(device))
                preds = np.rint(SX.numpy()).astype(np.int32)
                metrics = ClassificationAcc(preds, C.numpy().astype(np.int32), 2)
                print(f'test acc: {metrics.precision}\n{metrics.count_matrix}')
                # print(np.mean(SX.numpy()), np.std(SX.numpy()))
                torch.save(model, f'weights/model_gnn_edgedot_64_64_64_epoch{epoch}.pth')

                sample_size = 1000
                sample_size = min(sample_size, len(df_))
                sample_ids = np.random.choice(len(df_), min(len(df_), sample_size), replace=False)
                df = df_.iloc[sample_ids].copy()
                dataset = GraphDataset(df, feature_columns, 'cluster_id', 50, feature_divs=df_std)
                # dataset.cluster_transform(transforms=[GlobalJitterTransform(0.2), GlobalScaleTransform(0.5)])
                dataset.global_transform(transforms=[JitterTransform(0.05)])
                dataset.initialize()

        loss = train_epoch_step(epoch, dataset, model, optimizer, device)

        if (epoch) % 10 == 0:
            print(f'epoch {epoch}, loss: {loss}')









