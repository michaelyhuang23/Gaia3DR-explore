import sys
sys.path.append('..')

import torch
import seaborn as sns
import os
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from random import randint
import sklearn
import json
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import Planetoid
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD, Adam
from collections import Counter
from tools.neural_dataset import *
from tools.gnn_cluster import *
from tools.evaluation_metrics import ClassificationAcc, ClusterEvalIoU


if __name__ == '__main__':
    writer = SummaryWriter()
    np.random.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = Planetoid('data/Cora', name='Cora')
    X = dataset.data.x
    print(dataset.data)
    num_class = torch.max(dataset.data.y).item()+1
    print(num_class)
    EPOCH = 4000

    print(f'running with {device}')

    model = GCN_sample(X.shape[-1], num_class, device=device)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    # this is a different A, it contains the feature
    indices = dataset.data.edge_index.numpy()
    self_loops = np.stack([np.arange(X.shape[0]), np.arange(X.shape[0])])
    print(self_loops.shape)
    indices = np.concatenate([indices, indices[::-1,:]], axis=-1)
    indices = np.unique(indices, axis=-1)
    counts = np.bincount(indices.flatten()) // 2 # degree of each
    weights = np.ones((indices.shape[-1])) #
    weights /= np.sqrt(counts[indices[0]] * counts[indices[1]])
    D = torch.tensor(counts)
    A = torch.sparse_coo_tensor(indices, weights, (X.shape[0], X.shape[0])).float().coalesce()
    C = dataset.data.y
    print(A)
    A,X,C,D = A.to(device),X.to(device),C.to(device),D.to(device)

    def train_epoch_step(epoch, model, optimizer, mask, device):
        model.train()
        model.config(True)
        model.add_graph(D,A,X)
        FX = model(X)
        loss = F.cross_entropy(FX[mask], C[mask])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    for epoch in range(EPOCH):
        if (epoch) % 50 == 0:      
            with torch.no_grad():
                model.config(False)
                model.eval()
                model.add_graph(D,A,X)
                FX = model(X)
                preds = np.argmax(FX.numpy(), axis=-1).astype(np.int32)
                metrics = ClassificationAcc(preds[dataset.data.val_mask], C.cpu().numpy().astype(np.int32)[dataset.data.val_mask], num_class)
                print(f'train acc: {metrics.precision}, {metrics.recall}\n{metrics.count_matrix}')
                writer.add_scalar('Acc/train', metrics.precision, epoch)

        loss = train_epoch_step(epoch, model, optimizer, dataset.data.train_mask, device)

        if (epoch) % 50 == 0:
            print(f'epoch {epoch}, loss: {loss}')






