import torch
from torch import nn
import torch.nn.functional as F

class ClusterMap(nn.Module):
	def __init__(self, input_size, layer_sizes, device):
		super().__init__()
		self.device = device
		self.linears = nn.ModuleList()
		for i, size in enumerate(layer_sizes):
			prev_size = input_size if i==0 else layer_sizes[i-1]
			self.linears.append(nn.Linear(prev_size, size, device=self.device))
			if i < len(layer_sizes)-1:
				self.linears.append(nn.ReLU())

	def forward(self, X):
		return self.linears(X)

class ClassificationHead(nn.Module):
	def __init__(self, input_size, num_classes, device):
		super().__init__()
		self.classifier = nn.Linear(input_size, num_classes)
	
	def forward(self, X, y=None):
		'''
		X = (batch, features)
		y = (batch)
		'''
		X = self.classifier(X)
		if y == None:
			preds = torch.argmax(X, dim=-1)
			scores = F.softmax(X, dim=-1)
			return X, scores, preds, scores[preds]
		else:
			return F.cross_entropy(X, y)

class ClassificationModel(nn.Module):
	def __init__(self, input_size, num_classes, device, mapper = None, classifier = None):
		super().__init__()
		self.device = device
		self.mapper = mapper
		self.classifier = classifier
		if self.mapper == None:
			self.mapper = ClusterMap(input_size, [input_size, input_size, input_size], self.device)
		if self.classifier == None:
			self.classifier = ClassificationHead(input_size, num_classes, self.device)

	def forward(self, X, y=None):
		X = self.mapper(X)
		return self.classifier(X, y)


class PairwiseHead(nn.Module):
	def __init__(self, metric='euclidean'):
		self.metric = metric

	def forward(self, X, y):
		'''
		X = (batch, features),
		y = (batch)
		'''
		assert(y!=None)
		B = X.shape[0]
		L = X[None,...].repeat(B)
		R = X[:, None, :].repeat(1,B,1)
		if self.metric == 'euclidean':
			dist = torch.sqrt((L - R)**2, axis=-1)
		elif self.metric == 'manhattan':
			dist = torch.sum(torch.abs(L-R), axis=-1)
		else:
			raise ValueError('not fucking implemented')
		# what loss? naive loss
		Ly = y[None,...].repeat(B)
		Ry = y[...,None].repeat(1,B)
		loss = -torch.mean((Ly != Ry) * torch.log(dist+0.01))  # 0.01 is for stability of log
		return loss



