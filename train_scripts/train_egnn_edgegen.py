import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD, Adam
from collections import Counter
from torch.utils.tensorboard import SummaryWriter

from neural_dataset import *
from gnn_cluster import *
from evaluation_metrics import *
from cluster_analysis import C_HDBSCAN, C_GaussianMixture


if __name__ == '__main__':
    writer = SummaryWriter()
    np.random.seed(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCH = 4000
    data_root = 'data/simulation'
    dataset_name = 'm12i_cluster_data_large_cluster_v2'
    dataset_path = os.path.join(data_root, dataset_name)

    test_dataset_name = 'm12f_cluster_data_large_cluster_v2'
    test_dataset_path = os.path.join(data_root, test_dataset_name)

    print(f'running with {device}')

    df_ = pd.read_hdf(dataset_path+'.h5', key='star')
    with open(dataset_path+'_norm.json', 'r') as f:
        df_norm = json.load(f)
    df_test_ = pd.read_hdf(test_dataset_path+'.h5', key='star')
    with open(test_dataset_path+'_norm.json', 'r') as f:
        df_test_norm = json.load(f)

    df_norm['mean']['lzstar'] = 0
    df_test_norm['mean']['lzstar'] = 0
    df_norm['mean']['lxstar'] = 0
    df_test_norm['mean']['lxstar'] = 0
    df_norm['mean']['lystar'] = 0
    df_test_norm['mean']['lystar'] = 0
    df_norm['mean']['jzstar'] = 0
    df_test_norm['mean']['jzstar'] = 0
    df_norm['mean']['jrstar'] = 0
    df_test_norm['mean']['jrstar'] = 0

    sample_size = 1000
    sample_size = min(sample_size, len(df_test_))
    sample_ids = np.random.choice(len(df_test_), min(len(df_test_), sample_size), replace=False)
    df_test = df_test_.iloc[sample_ids].copy()

    sample_size = 1000
    sample_size = min(sample_size, len(df_))
    sample_ids = np.random.choice(len(df_), min(len(df_), sample_size), replace=False)
    df = df_.iloc[sample_ids].copy()

    print(df.columns)
    #feature_columns = ['estar', 'lzstar', 'lxstar', 'lystar', 'jzstar', 'jrstar', 'eccstar', 'rstar', 'feH', 'mgfe', 'xstar', 'ystar', 'zstar', 'vxstar', 'vystar', 'vzstar', 'vrstar', 'vphistar', 'vthetastar', 'omegaphistar', 'omegarstar', 'omegazstar', 'thetaphistar', 'thetarstar', 'thetazstar', 'zmaxstar']
    feature_columns = ['estar', 'feH', 'c_lzstar', 'jzstar', 'mgfe', 'vrstar', 'zstar', 'vphistar', 'eccstar']
    # feature_weights = np.array([0.71216923,0.555757,0.31106377,0.1477975,0.13819067,0.1145066,0.08163675,0.07823427,0.07169756])
    # feature_weights /= np.mean(feature_weights)
    feature_weights = np.ones((len(feature_columns)))

    dataset = GraphDataset(df, feature_columns, 'cluster_id', 999, normalize=False, feature_norms=df_norm, scales=feature_weights)
    dataset.initialize_dense(to_dense=True)

    test_dataset = GraphDataset(df_test, feature_columns, 'cluster_id', 999, normalize=False, feature_norms=df_test_norm, scales=feature_weights)
    test_dataset.initialize_dense(to_dense=True)

    model = GCNEdgeBasedEdgeGen(len(feature_columns), num_cluster=100, auxiliary=0, regularizer=0.00, device=device)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    def train_epoch_step(epoch, dataset, model, optimizer, device):
        model.train()
        model.config(True)
        A, X, _ = dataset[0]
        n = X.shape[0]
        D = dataset.D
        A,X,D = A.to(device),X.to(device),D.to(device)
        C = dataset.labels[None,...].repeat(n, 1) != dataset.labels[...,None].repeat(1, n)
        C = C.to(device)
        model.add_graph(D,A,X)
        model.add_connectivity(C)
        loss = model(X)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()


    for epoch in range(EPOCH):
        if (epoch) % 100 == 0:      
            with torch.no_grad():
                model.config(False)
                model.eval()
                A, X, _ = dataset[0]
                n = X.shape[0]
                D = dataset.D
                A,X,D = A.to(device),X.to(device),D.to(device)
                C = dataset.labels[None,...].repeat(n, 1) != dataset.labels[...,None].repeat(1, n)
                model.add_graph(D,A,X)
                FX, corr = model(X)
                FX = FX.detach().cpu().numpy()
                corr = corr.detach().cpu().numpy()
                metrics = ClusterEvalAll(FX, dataset.labels.numpy())
                print(metrics())

                preds = np.rint(corr).astype(np.int32).flatten()
                class_metrics = ClassificationAcc(preds, C.numpy().astype(np.int32).flatten(), 2)
                print(f'train acc: {class_metrics.precision}\n{class_metrics.count_matrix}')
                writer.add_scalar('Acc/train', class_metrics.precision, epoch)

                model.config(True)
                model.train()
                model.add_graph(D,A,X)
                C = dataset.labels[None,...].repeat(n, 1) != dataset.labels[...,None].repeat(1, n)
                C = C.to(device)
                model.add_connectivity(C)
                loss = model(X)
                print(f'train loss: {loss}')
                writer.add_scalar('Loss/train', loss, epoch)


                model.config(False)
                model.eval()
                A, X, _ = test_dataset[0]
                n = X.shape[0]
                D = test_dataset.D
                A,X,D = A.to(device),X.to(device),D.to(device)
                C = test_dataset.labels[None,...].repeat(n, 1) != test_dataset.labels[...,None].repeat(1, n)
                model.add_graph(D,A,X)
                FX, corr = model(X)
                FX = FX.detach().cpu().numpy()
                corr = corr.detach().cpu().numpy()
                metrics = ClusterEvalAll(FX, test_dataset.labels.numpy())
                print(metrics())

                preds = np.rint(corr).astype(np.int32).flatten()
                class_metrics = ClassificationAcc(preds, C.numpy().astype(np.int32).flatten(), 2)
                print(f'test acc: {class_metrics.precision}\n{class_metrics.count_matrix}')
                writer.add_scalar('Acc/test', class_metrics.precision, epoch)

                model.config(True)
                model.train()
                model.add_graph(D,A,X)
                C = test_dataset.labels[None,...].repeat(n, 1) != test_dataset.labels[...,None].repeat(1, n)
                C = C.to(device)
                model.add_connectivity(C)
                loss = model(X)
                print(f'test loss: {loss}')
                writer.add_scalar('Loss/test', loss, epoch)

                # print(np.mean(SX.numpy()), np.std(SX.numpy()))
                torch.save(model, f'weights/m12i_model_edgegen_32_32_epoch{epoch}.pth')

                sample_size = 1000
                sample_size = min(sample_size, len(df_test_))
                sample_ids = np.random.choice(len(df_test_), min(len(df_test_), sample_size), replace=False)
                df_test = df_test_.iloc[sample_ids].copy()
                test_dataset = GraphDataset(df_test, feature_columns, 'cluster_id', 999, normalize=False, feature_norms=df_test_norm, scales=feature_weights)
                test_dataset.initialize_dense(to_dense=True)

        if (epoch) % 10 == 0: 
            sample_size = 1000
            sample_size = min(sample_size, len(df_))
            sample_ids = np.random.choice(len(df_), min(len(df_), sample_size), replace=False)
            df = df_.iloc[sample_ids].copy()
            dataset = GraphDataset(df, feature_columns, 'cluster_id', 999, normalize=False, feature_norms=df_norm, scales=feature_weights)
            # dataset.global_transform(transforms=[JitterTransform(0.1)])
            dataset.initialize_dense(to_dense=True)

        loss = train_epoch_step(epoch, dataset, model, optimizer, device)

        if (epoch) % 1 == 0:
            print(f'epoch {epoch}, loss: {loss}')





