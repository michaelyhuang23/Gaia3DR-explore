from torch import nn
import torch.nn.functional as F
import torch
from scipy.optimize import linear_sum_assignment

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


class GNN(nn.Module): # non-overlapping
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


class GCNEdge(GNN): # non-overlapping
    def __init__(self, input_size, graph_layer_sizes=[32], linear_layer_sizes=[], similar_weight=1, device='cpu'):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.convs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i, size in enumerate(graph_layer_sizes):
            prev_size = input_size if i==0 else graph_layer_sizes[i-1]
            self.dropouts.append(nn.Dropout(p=0))
            self.convs.append(GCNConv(prev_size, size, device=self.device))
        self.linears = nn.ModuleList()
        for i, size in enumerate(linear_layer_sizes):
            prev_size = graph_layer_sizes[-1]*2 if i==0 else linear_layer_sizes[i-1]
            self.dropouts.append(nn.Dropout(p=0))
            self.linears.append(nn.Linear(prev_size, size, device=self.device))
        prev_size = graph_layer_sizes[-1]*2 if len(linear_layer_sizes)==0 else linear_layer_sizes[-1]
        self.dropouts.append(nn.Dropout(p=0))
        self.linears.append(nn.Linear(prev_size, 1))
        self.similar_weight = similar_weight

    def forward(self, X):
        c = 0
        for i,conv in enumerate(self.convs):
            # X = self.dropouts[c](X)
            # c+=1
            X = conv(self.A, X)
            X = F.relu(X)
        if not self.classify:
            print(torch.mean(X), torch.std(X))
        SX = torch.concat([X[self.A.indices()[0]], X[self.A.indices()[1]]], dim=-1)
        if not self.classify:
            print(torch.mean(SX), torch.std(SX))
        for i,linear in enumerate(self.linears):
            X = self.dropouts[c](X)
            c+=1
            SX = linear(SX)
            if i!=len(self.linears)-1:
                SX = F.relu(SX)
        if not self.classify:
            print(torch.mean(SX), torch.std(SX))
        SX = torch.sigmoid(SX)[:,0]
        if self.classify:
            weights = torch.ones_like(self.C, dtype=torch.float32)
            weights[SX<0.5] *= self.similar_weight
            return F.binary_cross_entropy(SX, self.C.float(), weight=weights)
        else:
            return SX

class GCNEdgeDot(GNN): # non-overlapping
    def __init__(self, input_size, graph_layer_sizes=[32], similar_weight=1, device='cpu'):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.convs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i, size in enumerate(graph_layer_sizes):
            prev_size = input_size if i==0 else graph_layer_sizes[i-1]
            self.dropouts.append(nn.Dropout(p=0))
            self.convs.append(GCNConv(prev_size, size, device=self.device))
        self.similar_weight = similar_weight

    def forward(self, X):
        c = 0
        for i,conv in enumerate(self.convs):
            X = self.dropouts[c](X)
            c+=1
            X = conv(self.A, X)
            X = F.relu(X)
        # if not self.classify:
        #     print(torch.mean(X), torch.std(X))
        SX = torch.sum(X[self.A.indices()[0]] * X[self.A.indices()[1]], dim=-1)
        # if not self.classify:
        #     print(torch.mean(SX), torch.std(SX))
        SX = torch.sigmoid(SX)
        if self.classify:
            weights = torch.ones_like(self.C, dtype=torch.float32)
            weights[SX<0.5] *= self.similar_weight
            return F.binary_cross_entropy(SX, self.C.float(), weight=weights)
        else:
            return SX

class NodeGCNConv(nn.Module): # has relu built in
    def __init__(self, input_channels_n, input_channels_e, output_channels_n, activation=True, device='cpu'):
        super().__init__()
        self.device = device
        self.pass_map = nn.Linear(input_channels_e, output_channels_n, device=device)
        self.self_map = nn.Linear(input_channels_n, output_channels_n, device=device)
        self.activation = activation

    def forward(self, D, A, X):
        '''
        D is a dense tensor (n)
        A is a sparse tensor (n, n, input_channels_e)
        X is a dense tensor (n, input_channels_n)
        '''
        if A.is_sparse:
            A = torch.sparse.sum(A, dim=1).to_dense()
        else:
            A = torch.sum(A, dim=1)
        if self.activation:
            return F.relu(self.pass_map(A/D[...,None]) + self.self_map(X)) # row normalized
        else:
            return self.pass_map(A/D[...,None]) + self.self_map(X) # row normalized

class EdgeGCNConv(nn.Module): # has relu built in
    def __init__(self, input_channels_n, input_channels_e, output_channels_e, activation=True, device='cpu'):
        super().__init__()
        self.device = device
        self.pass_map = nn.Linear(input_channels_n*2, output_channels_e, device=device)
        self.self_map = nn.Linear(input_channels_e, output_channels_e, device=device)
        self.output_size = output_channels_e
        self.activation = activation

    def forward(self, A, X):
        '''
        A is a sparse tensor (n, n, input_channels_e)
        X is a dense tensor (n, input_channels_n)
        '''
        if A.is_sparse:
            X1, X2 = X[A.indices()[0]], X[A.indices()[1]]
            X1, X2 = (X1-X2)/2, (X1+X2)/2
            E = torch.concat([X1, X2], dim=-1)
            if self.activation:
                nA = torch.sparse_coo_tensor(A.indices(), F.relu(self.pass_map(E) + self.self_map(A.values())), (*A.shape[:2], self.output_size)).float().coalesce()
            else:
                nA = torch.sparse_coo_tensor(A.indices(), self.pass_map(E) + self.self_map(A.values()), (*A.shape[:2], self.output_size)).float().coalesce()
            return nA
        else:
            n = X.shape[0]
            X1, X2 = X[None,...].repeat(n,1,1), X[:,None,:].repeat(1,n,1)
            X1, X2 = (X1-X2)/2, (X1+X2)/2
            E = torch.concat([X1, X2], dim=-1)
            nA = self.pass_map(E) + self.self_map(A)
            if self.activation:
                nA = F.relu(nA)
            return nA

class GCNEdgeBased(GNN): # non-overlapping
    def __init__(self, input_size, similar_weight=1, device='cpu'):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.convN1 = NodeGCNConv(input_size, input_size, 32)
        self.dropout1 = nn.Dropout(p=0.0)
        self.convE1 = EdgeGCNConv(32, input_size, 32)
        self.convN2 = NodeGCNConv(32, 32, 32)
        self.dropout2 = nn.Dropout(p=0.0)
        self.convE2 = EdgeGCNConv(32, 32, 32)
        self.classifier = nn.Linear(32, 1)
        self.similar_weight = similar_weight

    def add_graph(self, D, A, X):
        self.D = D
        X1, X2 = X[A.indices()[0]], X[A.indices()[1]]
        weights = torch.abs(X1-X2)
        self.A = torch.sparse_coo_tensor(A.indices(), weights, (*A.shape, weights.shape[-1])).coalesce().float()

    def forward(self, X):
        X = torch.zeros_like(X)
        A = self.A.clone()
        X = self.convN1(self.D, A, X)
        X = self.dropout1(X)
        A = self.convE1(A, X)
        X = self.convN2(self.D, A, X)
        X = self.dropout2(X)
        A = self.convE2(A, X)
        # assert torch.all(A.indices() == self.A.indices()).item()
        SX = self.classifier(A.values())
        SX = torch.sigmoid(SX)[:,0]
        if self.classify:
            weights = torch.ones_like(self.C, dtype=torch.float32)
            weights[SX<0.5] *= self.similar_weight
            return F.binary_cross_entropy(SX, self.C.float(), weight=weights)
        else:
            return SX

class FakeGNN(GNN): # non-overlapping
    def __init__(self, input_size, similar_weight=1, device='cpu'):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.classifier = nn.Linear(input_size, 1)
        self.similar_weight = similar_weight

    def add_graph(self, D, A, X):
        self.D = D
        X1, X2 = X[A.indices()[0]], X[A.indices()[1]]
        weights = torch.abs(X1-X2)
        self.A = torch.sparse_coo_tensor(A.indices(), weights, (*A.shape, weights.shape[-1])).coalesce().float()

    def forward(self, X):
        A = self.A
        SX = self.classifier(A.values())
        SX = torch.sigmoid(SX)[:,0]
        if self.classify:
            weights = torch.ones_like(self.C, dtype=torch.float32)
            weights[SX<0.5] *= self.similar_weight
            return F.binary_cross_entropy(SX, self.C.float(), weight=weights)
        else:
            return SX

class GCNEdge2Cluster(GNN): # non-overlapping
    def __init__(self, input_size, num_cluster=30, graph_layer_sizes=[32], regularizer=0.01, device='cpu'):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.convs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i, size in enumerate(graph_layer_sizes):
            prev_size = input_size if i==0 else graph_layer_sizes[i-1]
            self.dropouts.append(nn.Dropout(p=0))
            self.convs.append(GCNConv(prev_size, size, device=self.device))
        prev_size = graph_layer_sizes[-1] if len(graph_layer_sizes)!=0 else input_size
        self.dropouts.append(nn.Dropout(p=0))
        self.convs.append(GCNConv(prev_size, num_cluster, device=self.device))
        self.num_cluster = num_cluster
        self.regularizer = regularizer

    def forward(self, X):
        c = 0
        for i,conv in enumerate(self.convs):
            X = self.dropouts[c](X)
            c+=1
            X = conv(self.A, X)
            if i != len(self.convs)-1:
                X = F.relu(X)
        FX = F.softmax(X, dim=-1) # change to softmax if not doing overlapping clusters
        FF = torch.sum(FX[self.A.indices()[0]] * FX[self.A.indices()[1]], dim=-1)
        NFX = torch.log(1-FX**2)
        pregularize = -torch.sum(torch.log(1.0001-torch.exp(torch.sum(NFX, dim=0))), dim=0)
        if self.classify:
            loss = torch.mean((FF - self.C)**2)
            print(loss.item(), self.regularizer*pregularize.item())
            return loss + self.regularizer * pregularize
        else:
            return FX

class GCNEdgeBasedEdgeGen(GNN): # non-overlapping
    def __init__(self, input_size, num_cluster=30, auxiliary=1, regularizer=0.0001, similar_weight=1, device='cpu'):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.convN1 = NodeGCNConv(input_size, input_size, 32)
        self.dropout1 = nn.Dropout(p=0.0)
        self.convE1 = EdgeGCNConv(32, input_size, 32)
        self.convN2 = NodeGCNConv(32, 32, 32)
        self.dropout2 = nn.Dropout(p=0.0)
        self.convE2 = EdgeGCNConv(32, 32, 32)
        self.convN3 = NodeGCNConv(32, 32, num_cluster, activation=False)
        self.classifier = nn.Linear(32, 1)
        self.auxiliary = auxiliary
        self.regularizer = regularizer
        self.similar_weight = similar_weight

    def add_graph(self, D, A, X):
        self.D = D
        if A.is_sparse:
            X1, X2 = X[A.indices()[0]], X[A.indices()[1]]
            weights = torch.abs(X1-X2)
            self.A = torch.sparse_coo_tensor(A.indices(), weights, (*A.shape, weights.shape[-1]), device=self.device).coalesce().float()
        else:
            n = X.shape[0]
            X1, X2 = X[None,...].repeat(n, 1, 1), X[:,None,:].repeat(1, n, 1)
            self.A = torch.abs(X1 - X2) # change to decreasing function

    def forward(self, X):
        X = torch.zeros_like(X, device=self.device)
        A = self.A.clone().to(self.device)
        X = self.convN1(self.D, A, X)
        X = self.dropout1(X)
        A = self.convE1(A, X)
        X = self.convN2(self.D, A, X)
        X = self.dropout2(X)
        A = self.convE2(A, X)
        if self.classify:
            if not A.is_sparse:
                SA = torch.sigmoid(self.classifier(A))[:,:,0]
                SA = torch.clip(SA, 0.001, 0.99)
                loss_class = F.binary_cross_entropy(SA, self.C.float())
            else:
                raise ValueError('not fucking implemented')
        FX = self.convN3(self.D, A, X)
        FX = torch.softmax(FX, dim=-1)
        NFX = torch.log(1-FX**2*0.9999)
        pregularize = -torch.sum(torch.log(1-torch.exp(torch.sum(NFX, dim=0))*0.9999), dim=0)
        corr = 1-torch.mm(FX, torch.transpose(FX, 0, 1))
        if self.classify:
            loss_gen = F.binary_cross_entropy(corr.flatten(), self.C.flatten().float())
            print(loss_gen.item(), loss_class.item()*self.auxiliary, pregularize.item()*self.regularizer)
            return loss_gen + loss_class*self.auxiliary + pregularize*self.regularizer
        else:
            return FX, corr


class GCNEdgeBasedCluster(GNN): # non-overlapping
    def __init__(self, input_size, num_cluster=30, similar_weight=1, device='cpu'):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.convN1 = NodeGCNConv(input_size, input_size, 32)
        self.convE1 = EdgeGCNConv(32, input_size, 32)
        self.convN2 = NodeGCNConv(32, 32, 32)
        self.convE2 = EdgeGCNConv(32, 32, 32)
        self.convN3 = NodeGCNConv(32, 32, num_cluster, activation=False)
        self.similar_weight = similar_weight

    def add_graph(self, D, A, X):
        self.D = D
        X1, X2 = X[A.indices()[0]], X[A.indices()[1]]
        weights = torch.abs(X1-X2)
        self.A = torch.sparse_coo_tensor(A.indices(), weights, (*A.shape, weights.shape[-1])).coalesce().float()

    def forward(self, X):
        X = torch.zeros_like(X)
        A = self.A.clone()
        X = self.convN1(self.D, A, X)
        A = self.convE1(A, X)
        X = self.convN2(self.D, A, X)
        A = self.convE2(A, X)
        FX = self.convN3(self.D, A, X)
        FX = torch.softmax(FX, dim=-1)

        if self.classify:
            SX = torch.log(FX)
            corr = -torch.mm(torch.transpose(self.C.float(),0,1), SX)
            with torch.no_grad():
                label_idx, pred_idx = linear_sum_assignment(corr.detach().numpy())
            loss = torch.mean(corr[label_idx, pred_idx])
            return loss
        else:
            return FX

class GCNEdgeBasedEdgeGenCluster(GNN): # non-overlapping
    def __init__(self, input_size, num_cluster=30, loss_weights=[1,1,1,0.1], device='cpu'):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.convN1 = NodeGCNConv(input_size, input_size, 32)
        self.dropout1 = nn.Dropout(p=0.3)
        self.convE1 = EdgeGCNConv(32, input_size, 32)
        self.convN2 = NodeGCNConv(32, 32, 32)
        self.dropout2 = nn.Dropout(p=0.3)
        self.convE2 = EdgeGCNConv(32, 32, 32)
        self.convN3 = NodeGCNConv(32, 32, num_cluster, activation=False)
        self.classifier = nn.Linear(32, 1)
        self.num_cluster = num_cluster
        self.loss_weights = loss_weights

    def add_graph(self, D, A, X):
        self.D = D
        X1, X2 = X[A.indices()[0]], X[A.indices()[1]]
        weights = torch.abs(X1-X2)
        self.A = torch.sparse_coo_tensor(A.indices(), weights, (*A.shape, weights.shape[-1])).coalesce().float()

    def add_connectivity(self, C): # let C be the assignment matrix in this case
        self.C = C
        CI = torch.argmax(C, dim=-1)
        self.CE = (CI[self.A.indices()[0]] != CI[self.A.indices()[1]]).float()

    def forward(self, X):
        X = torch.zeros_like(X)
        A = self.A.clone()
        X = self.convN1(self.D, A, X)
        X = self.dropout1(X)
        A = self.convE1(A, X)

        if self.classify:
            SX = self.classifier(A.values())
            SX = torch.sigmoid(SX)[:,0]
            loss_edge_class = F.binary_cross_entropy(SX, self.CE)

        X = self.convN2(self.D, A, X)
        X = self.dropout2(X)
        A = self.convE2(A, X)
        FX = self.convN3(self.D, A, X)
        FX = torch.softmax(FX, dim=-1)

        if self.classify:
            FF = torch.sum(FX[self.A.indices()[0]] * FX[self.A.indices()[1]], dim=-1)
            loss_edge_regen = torch.mean((FF - (1-SX))**2) * self.num_cluster**2
            regularize = torch.sqrt(torch.sum(torch.sum(FX, dim=0)**2))*(self.num_cluster**0.5)/X.shape[0] - 1 # this is frobenius norm regularization

        if self.classify:
            LX = torch.log(FX+0.0001)
            corr = -torch.mm(torch.transpose(self.C.float(),0,1), LX)
            with torch.no_grad():
                label_idx, pred_idx = linear_sum_assignment(corr.detach().numpy())
            loss_cluster = torch.sum(corr[label_idx, pred_idx])/len(X)

        if self.classify:
            print(loss_edge_class.item() * self.loss_weights[0], loss_edge_regen.item() * self.loss_weights[1], loss_cluster.item() * self.loss_weights[2], regularize.item() * self.loss_weights[3])
            loss = loss_edge_class * self.loss_weights[0] + loss_edge_regen * self.loss_weights[1] + loss_cluster * self.loss_weights[2] + regularize * self.loss_weights[3]
            return loss, loss_edge_class * self.loss_weights[0], loss_edge_regen * self.loss_weights[1], loss_cluster * self.loss_weights[2], regularize * self.loss_weights[3]
        else:
            return FX

        

