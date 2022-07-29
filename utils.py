import numpy as np
import pandas as pd
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

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
