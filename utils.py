import numpy as np
import pandas as pd
import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 5000, device='cpu'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:, :(pe.shape[1]-1)//2]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class LabelEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, device='cpu'):
        super().__init__()
        min_F, max_F = 1/d_model, 1/3
        W = torch.linspace(min_F, max_F, max_len) * 2 * math.pi
        positions = torch.arange(d_model)
        self.encoding = torch.cos(W[:,None] * positions[None,:])

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, embedding_dim]
        """
        return x + self.encoding[:x.size(0)]

def cart2spherical(x, y, z):
    r = np.linalg.norm([x,y,z], axis=0)
    rho = np.linalg.norm([x,y], axis=0)
    phi = np.arctan2(y, x)
    theta = np.arctan2(rho, z)
    return r, phi, theta

class UnionFind:
    def __init__(self, n):
        self.parents = list(range(n))

    def find(self, a):
        if a == self.parents[a] : return a
        pa = self.find(self.parents[a])
        self.parents[a] = pa
        return pa

    def join(self, a, b):
        pa = self.find(a)
        pb = self.find(b)
        if pa != pb:
            self.parents[pa] = pb

    def connect(self, a, b):
        return self.find(a) == self.find(b)
