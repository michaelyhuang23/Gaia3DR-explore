import torch
from torch import nn
import torch.nn.functional as F


class ScaleContrastModel(nn.Module):
    def __init__(self, input_size, device='cpu'):
        super().__init__()
        self.device = device
        self.classifier = nn.Linear(input_size, 1, device=self.device)

    def config(self, classify=True):
        self.classify = classify

    def forward(self, X1, X2, y1=None, y2=None):
        X = torch.abs(X1-X2)
        X = torch.sigmoid(self.classifier(X))[:,0]
        if y1 is not None and y2 is not None and self.classify:
            return F.binary_cross_entropy(X, (y1!=y2).float())
        else:
            return X

class ContrastModel(nn.Module):
    def __init__(self, input_size, layer_sizes, similar_weight=1, device='cpu'):
        super().__init__()
        self.device = device
        self.linears = nn.ModuleList()
        for i, size in enumerate(layer_sizes):
            prev_size = input_size*2 if i==0 else layer_sizes[i-1]
            self.linears.append(nn.Linear(prev_size, size, device=self.device))
        self.classifier = nn.Linear(layer_sizes[-1], 1, device=self.device)
        self.similar_weight = similar_weight

    def config(self, classify=True):
        self.classify = classify

    def forward(self, X1, X2, y1=None, y2=None):
        X = torch.concat([X1-X2, (X1+X2)/2], dim=-1)
        for i,linear in enumerate(self.linears):
            X = linear(X)
            X = F.relu(X)
        X = torch.sigmoid(self.classifier(X))[:,0]
        if y1 is not None and y2 is not None and self.classify:
            weights = torch.ones_like(y1)
            weights[X<0.5] *= self.similar_weight
            return F.binary_cross_entropy(X, (y1!=y2).float(), weight=weights)
        else:
            return X


class ClusterMap(nn.Module):
    def __init__(self, input_size, layer_sizes, device='cpu'):
        super().__init__()
        self.device = device
        self.linears = nn.ModuleList()
        for i, size in enumerate(layer_sizes):
            prev_size = input_size if i==0 else layer_sizes[i-1]
            self.linears.append(nn.Linear(prev_size, size, device=self.device))
        self.output_size = layer_sizes[-1]

    def forward(self, X):
        for i,linear in enumerate(self.linears):
            X = linear(X)
            if i != len(self.linears)-1:
                X = F.relu(X)
        return X

class ClassificationHead(nn.Module):
    def __init__(self, input_size, num_classes, weights=None, device='cpu'):
        super().__init__()
        self.device = device
        self.classifier = nn.Linear(input_size, num_classes, device=self.device)
        self.weights = weights
    
    def forward(self, X, y=None):
        '''
        X = (batch, features)
        y = (batch)
        '''
        X = self.classifier(X)
        if y == None:
            preds = torch.argmax(X, dim=-1)
            scores = F.softmax(X, dim=-1)
            return X, scores, preds, scores.gather(-1, preds[...,None]).squeeze()
        else:
            if self.weights is None:
                return F.cross_entropy(X, y)
            else:
                return F.cross_entropy(X, y, weight=self.weights)

class SimpleGaussianHead(nn.Module): # this can be interpreted as an adversarial gaussian mixture module
    def __init__(self, input_size, num_classes, weights=None, regularize=0.1, device='cpu'):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.W = nn.Parameter(torch.rand(self.num_classes, self.input_size)*2-1) # centered at 0, with spread=1
        self.W.requires_grad = True
        self.weights = weights
        self.regularize = regularize
    
    def forward(self, X, y=None):
        '''
        X = (batch, features)
        y = (batch)
        '''
        avg_dist = torch.mean(X**2)
        X = -torch.mean((X[:,None,:] - self.W[None,...])**2, dim=-1) # mean standardizes distance, standard distance in kd space is sqrt(k)*sigma
        # applying crossentropy loss on this is equivalent to standard variance gaussian assumptions
        if y == None:
            preds = torch.argmax(X, dim=-1)
            scores = F.softmax(X, dim=-1)
            return X, scores, preds, scores.gather(-1, preds[...,None]).squeeze()
        else:
            if self.weights is None:
                return F.cross_entropy(X, y) + self.regularize*avg_dist
            else:
                return F.cross_entropy(X, y, weight=self.weights) + self.regularize*avg_dist

class GaussianHead(nn.Module): 
    def __init__(self, input_size, num_classes, weights=None, regularize=0.1, device='cpu'):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.W = nn.Parameter(torch.rand(self.num_classes, self.input_size)*2-1) # centered at 0, with spread=1
        self.W.requires_grad = True
        self.weights = weights
        self.regularize = regularize
    
    def forward(self, X, y=None):
        '''
        X = (batch, features)
        y = (batch)
        '''
        avg_dist = torch.mean(X**2)
        X = -torch.mean((X[:,None,:] - self.W[None,...])**2, dim=-1) # mean standardizes distance, standard distance in kd space is sqrt(k)*sigma
        # applying crossentropy loss on this is equivalent to standard variance gaussian assumptions
        if y == None:
            preds = torch.argmax(X, dim=-1)
            scores = F.softmax(X, dim=-1)
            return X, scores, preds, scores.gather(-1, preds[...,None]).squeeze()
        else:
            if self.weights is None:
                return F.cross_entropy(X, y) + self.regularize*avg_dist
            else:
                return F.cross_entropy(X, y, weight=self.weights) + self.regularize*avg_dist


class ClassificationModel(nn.Module):
    def __init__(self, input_size, num_classes, device='cpu', mapper = None, classifier = None):
        super().__init__()
        self.device = device
        self.mapper = mapper
        self.classifier = classifier
        if self.mapper == None:
            self.mapper = ClusterMap(input_size, [input_size, input_size, input_size], self.device)
        if self.classifier == None:
            self.classifier = ClassificationHead(self.mapper.output_size, num_classes, self.device)

    def config(self, classify=True):
        self.classify = classify

    def forward(self, X, y=None):
        X = self.mapper(X)
        if self.classify:
            return self.classifier(X, y)
        else:
            return X


class PairwiseHead(nn.Module):
    def __init__(self, metric='euclidean', regularize=0.01):
        super().__init__()
        self.metric = metric
        self.regularize = regularize

    def forward(self, X, y):
        '''
        X = (batch, features),
        y = (batch)
        '''
        B = X.shape[0]
        avg_dist = torch.mean(X**2)
        L = X[None, ...]
        R = X[:, None, :]
        if self.metric == 'euclidean':
            dist = torch.sum((L - R)**2, dim=-1) 
        elif self.metric == 'manhattan':
            dist = torch.sum(torch.abs(L-R), axis=-1)
        else:
            raise ValueError('not fucking implemented')
        
        Ly = y[None,...].repeat(B,1)
        Ry = y[...,None].repeat(1,B)
        loss = - torch.mean((Ly != Ry) * torch.log(1.01-torch.exp(-dist))) - torch.mean((Ly == Ry) * torch.log(torch.exp(-dist)))  # 0.01 is for stability of log
        loss += self.regularize * avg_dist
        return loss

class PairwiseModel(nn.Module):
    def __init__(self, input_size, metric='euclidean', device='cpu', mapper = None, pairloss = None):
        super().__init__()
        self.device = device
        self.mapper = mapper
        self.pairloss = pairloss
        if self.mapper == None:
            self.mapper = ClusterMap(input_size, [input_size, input_size, input_size], self.device)
        if self.pairloss == None:
            self.pairloss = PairwiseHead(metric)

    def config(self, pair=True):
        self.pair = pair

    def forward(self, X, y=None):
        X = self.mapper(X)
        if y is None or self.pair == False:
            return X
        elif self.pair == True:
            return self.pairloss(X, y)
        else:
            raise ValueError('parameter not fucking set correctly')



