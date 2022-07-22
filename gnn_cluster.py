from torch import nn
import torch.nn.functional as F

class GCNConv(nn.Module):
	def __init__(self, input_channels, output_channels, device='cpu'):
		self.device = device
		self.pass_map = nn.Linear(input_channels, output_channels, device=device)
		self.self_map =  nn.Linear(input_channels, output_channels, device=device)

	def forward(self, A, X):
		'''
		A is a sparse tensor
		X is a dense tensor
		'''
		return F.relu(self.pass_map(torch.sparse.mm(A, X)) + self.self_map(X))

class GCNCluster(nn.Module): # non-overlapping
	def __init__(self, input_size, num_clusters, graph_layer_sizes=[16,16], linear_layer_sizes=[], device='cpu'):
		super().__init__()
		self.device = device
		self.input_size = input_size
		self.num_clusters = num_clusters
		self.convs = nn.ModuleList()
		for i, size in enumerate(graph_layer_sizes):
			prev_size = input_size if i==0 else graph_layer_sizes[i-1]
			self.convs.append(GCNConv(prev_size, size, device=self.device))
		self.linears = nn.ModuleList()
		for i, size in enumerate(linear_layer_sizes):
			prev_size = graph_layer_sizes[-1] if i==0 else linear_layer_sizes[i-1]
			self.linears.append(nn.Linear(prev_size, size, device=self.device))
		prev_size = graph_layer_sizes[-1] if len(linear_layer_sizes)==0 else linear_layer_sizes[-1]
		self.linears.append(nn.Linear(prev_size, num_clusters))

	def forward(self, data):
		X, edge_index = data.x, data.edge_index
		for i,conv in enumerate(self.convs):
			X = conv(X, edge_index)
			X = F.relu(X)
		for i,linear in enumerate(self.linears):
			X = linear(X)
			if i!=len(self.linears)-1:
				X = F.relu(X)
		S = F.softmax(X, dim=-1)
		if self.training:

		else:
			return S