from torch import nn
import torch.nn.functional as F
import torch

class GCNConv(nn.Module):
	def __init__(self, input_channels, output_channels, device='cpu'):
		super().__init__()
		self.device = device
		self.pass_map = nn.Linear(input_channels, output_channels, device=device)
		self.self_map =  nn.Linear(input_channels, output_channels, device=device)

	def forward(self, A, X):
		'''
		A is a sparse tensor
		X is a dense tensor
		'''
		return self.pass_map(torch.sparse.mm(A, X)) + self.self_map(X)

class GCNEdge(nn.Module): # non-overlapping
	def __init__(self, input_size, graph_layer_sizes=[32,32], linear_layer_sizes=[32], similar_weight=1, device='cpu'):
		super().__init__()
		self.device = device
		self.input_size = input_size
		self.convs = nn.ModuleList()
		for i, size in enumerate(graph_layer_sizes):
			prev_size = input_size if i==0 else graph_layer_sizes[i-1]
			self.convs.append(GCNConv(prev_size, size, device=self.device))
		self.linears = nn.ModuleList()
		for i, size in enumerate(linear_layer_sizes):
			prev_size = graph_layer_sizes[-1]*2 if i==0 else linear_layer_sizes[i-1]
			self.linears.append(nn.Linear(prev_size, size, device=self.device))
		prev_size = graph_layer_sizes[-1] if len(linear_layer_sizes)==0 else linear_layer_sizes[-1]
		self.linears.append(nn.Linear(prev_size, 1))
		self.similar_weight = similar_weight

	def add_graph(self, A):
		'''
		Adj matrix in sparse tensor form
		'''
		self.A = A

	def add_connectivity(self, C):
		'''
		1D vector denoting if the edge disconnected
		'''
		self.C = C

	def config(self, classify=True):
		self.classify = classify

	def forward(self, X):
		for i,conv in enumerate(self.convs):
			X = conv(self.A, X)
			X = F.relu(X)
		SX = torch.concat([X[self.A.indices()[0]], X[self.A.indices()[1]]], dim=-1)
		for i,linear in enumerate(self.linears):
			SX = linear(SX)
			if i!=len(self.linears)-1:
				SX = F.relu(SX)
		SX = torch.sigmoid(SX)[:,0]
		if self.classify:
			weights = torch.ones_like(self.C, dtype=torch.float32)
			weights[SX<0.5] *= self.similar_weight
			return F.binary_cross_entropy(SX, self.C.float(), weight=weights)
		else:
			return SX

