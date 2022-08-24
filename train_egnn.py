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
from evaluation_metrics import ClassificationAcc, ClusterEvalIoU
from cluster_analysis import C_HDBSCAN, C_GaussianMixture


if __name__ == '__main__':
    writer = SummaryWriter()
    np.random.seed(0)
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
    df_test = sample_space(df_test_, radius=5, radius_sun=8.2, sample_size=sample_size)

    sample_size = 1000
    df = sample_space(df_, radius=5, radius_sun=8.2, sample_size=sample_size)

    print(df.columns)
    # feature_columns = ['estar', 'lzstar', 'lxstar', 'lystar', 'jzstar', 'jrstar', 'eccstar', 'rstar', 'feH', 'mgfe', 'zstar', 'vrstar', 'vphistar', 'vthetastar', 'omegaphistar', 'omegarstar', 'omegazstar', 'thetaphistar', 'thetarstar', 'thetazstar', 'zmaxstar']
    # feature_columns = ['feH', 'estar', 'lzstar', 'eccstar', 'vthetastar', 'thetazstar', 'jrstar', 'thetaphistar', 'vrstar', 'lystar']
    # feature_columns = ['estar', 'feH', 'lzstar', 'jzstar', 'mgfe', 'vrstar', 'zstar', 'vphistar', 'eccstar']
    feature_columns = ['estar', 'feH', 'lzstar', 'lystar', 'lxstar', 'jzstar', 'jrstar', 'mgfe','eccstar', 'zstar']
    
    # feature_weights = np.array([0.71216923,0.555757,0.31106377,0.1477975,0.13819067,0.1145066,0.08163675,0.07823427,0.07169756])
    # feature_weights /= np.mean(feature_weights)
    feature_weights = np.ones((len(feature_columns)))

    dataset = GraphDataset(df, feature_columns, 'cluster_id', 999, normalize=False, feature_norms=df_norm, scales=feature_weights)
    dataset.initialize_dense()

    test_dataset = GraphDataset(df_test, feature_columns, 'cluster_id', 999, normalize=False, feature_norms=df_test_norm, scales=feature_weights)
    test_dataset.initialize_dense()

    print(torch.mean(dataset.features, axis=0))
    print(torch.mean(test_dataset.features, axis=0))

    model = GCNEdgeBased(len(feature_columns), similar_weight=1, device=device)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    def train_epoch_step(epoch, dataset, model, optimizer, device):
        model.train()
        model.config(True)
        A, X, C = dataset[0]
        D = dataset.D
        A,X,C,D = A.to(device),X.to(device),C.to(device),D.to(device)
        model.add_graph(D,A,X)
        model.add_connectivity(C)
        loss = model(X)
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
                D = dataset.D
                A,X,C,D = A.to(device),X.to(device),C.to(device),D.to(device)
                model.add_graph(D,A,X)
                model.add_connectivity(C)
                SX = model(X).cpu()
                preds = np.rint(SX.numpy()).astype(np.int32)
                metrics = ClassificationAcc(preds, C.cpu().numpy().astype(np.int32), 2)
                print(f'train acc: {metrics.precision}\n{metrics.count_matrix}')
                writer.add_scalar('Acc/train', metrics.precision, epoch)

                model.config(True)
                model.train()
                model.add_graph(D,A,X)
                model.add_connectivity(C)
                loss = model(X)
                print(f'train loss: {loss}')
                writer.add_scalar('Loss/train', loss, epoch)

                avg_loss = np.zeros((10))
                avg_precision = np.zeros((10))
                for eval_run in range(10):
                    sample_size = 1000
                    df_test = sample_space(df_test_, radius=5, radius_sun=8.2, sample_size=sample_size)
                    test_dataset = GraphDataset(df_test, feature_columns, 'cluster_id', 999, normalize=False, feature_norms=df_test_norm, scales=feature_weights)
                    test_dataset.initialize_dense()

                    model.config(False)
                    model.eval()
                    A, X, C = test_dataset[0]
                    D = test_dataset.D
                    A,X,C,D = A.to(device),X.to(device),C.to(device),D.to(device)
                    model.add_graph(D,A,X)
                    model.add_connectivity(C)
                    SX = model(X).cpu()
                    preds = np.rint(SX.numpy()).astype(np.int32)
                    metrics = ClassificationAcc(preds, C.cpu().numpy().astype(np.int32), 2)
                    print(f'test acc: {metrics.precision}\n{metrics.count_matrix}')
                    avg_precision[eval_run] = metrics.precision

                    model.config(True)
                    model.train()
                    model.add_graph(D,A,X)
                    model.add_connectivity(C)
                    loss = model(X)
                    print(f'test loss: {loss}')
                    avg_loss[eval_run] = loss

                print(f'avg test loss: {np.mean(avg_loss)}')
                print(f'avg test precision: {np.mean(avg_precision)}')
                writer.add_scalar('Acc/test', np.mean(avg_precision), epoch)
                writer.add_scalar('Loss/test', np.mean(avg_loss), epoch)

                # print(np.mean(SX.numpy()), np.std(SX.numpy()))
                torch.save(model, f'weights/m12i_dense_orig_model_32_32_epoch{epoch}.pth')

        if epoch % 5 == 0:
            sample_size = 1000
            df = sample_space(df_, radius=5, radius_sun=8.2, sample_size=sample_size)
            dataset = GraphDataset(df, feature_columns, 'cluster_id', 999, normalize=False, feature_norms=df_norm, scales=feature_weights)
            # dataset.cluster_transform(transforms=[GlobalJitterTransform(0.2), GlobalScaleTransform(0.5)])
            # dataset.global_transform(transforms=[JitterTransform(0.05)])
            dataset.initialize_dense()

        loss = train_epoch_step(epoch, dataset, model, optimizer, device)

        if (epoch) % 1 == 0:
            print(f'epoch {epoch}, loss: {loss}')





