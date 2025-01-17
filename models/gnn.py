# modules
import torch
import numpy as np
from itertools import product
import torch.nn as nn
# The wn is a fun of normalization for param weight matrix
from torch.nn.utils import weight_norm as wn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv,GATConv,SAGEConv,GINConv,EdgeConv,TAGConv,GINEConv
from sklearn.linear_model import LogisticRegression, OrthogonalMatchingPursuit
from models import mlp
from copy import deepcopy
from models import utils as ut
EPS = 1e-30

def classify(feats, y=None, do_mask=True, omp_rew=None):
    def check_coef(coef, intercept):
        best_coef = deepcopy(coef)
        best_intercept = deepcopy(intercept)
        if best_coef.shape[0] == 1:
            best_coef = np.concatenate((-best_coef/2, best_coef/2), axis=0)
            best_intercept = np.array([-best_intercept[0]/2, best_intercept[0]/2])
        return best_coef, best_intercept

    if y is None:
        y = np.arange(len(feats))
    if do_mask:
        clf=LogisticRegression(max_iter=200).fit(feats,y)
        best_coef, best_intercept = check_coef(clf.coef_, clf.intercept_) # (n classes, n features)  
        return best_coef, best_intercept  
    else:
        assert omp_rew
        best_rew, best_coef_intercept, best_set_indicator = -1e9, None, None
        for k in range(1, len(feats)+1):
            omp=OrthogonalMatchingPursuit(n_nonzero_coefs=k,
                                    normalize=True, fit_intercept=True).fit(feats, np.eye(len(feats)))
            set_indices = np.argsort((omp.coef_!=0).sum(axis=0))[-k:].tolist() # choose top k nodes
            set_indicator = np.zeros(len(feats))
            set_indicator[set_indices] = 1.
            rew = omp_rew(set_indicator[None], omp.score(feats, np.eye(len(feats))))
            if rew > best_rew:
                best_set_indicator = deepcopy(set_indicator)
                best_rew = rew
                best_coef_intercept = (deepcopy(omp.coef_), deepcopy(omp.intercept_))
        
        best_coef, best_intercept = check_coef(*best_coef_intercept)
        return best_coef, best_intercept, best_set_indicator

    
            




#MDP baseline with distances as final hidden representation
class DistMask(nn.Module):
    def __init__(self, output_dim, mask_c=.0,do_omp=None):
        super(DistMask,self).__init__()   
        self.output_dim = output_dim
        self.do_mask = mask_c > .0
        self.do_omp = bool(do_omp)
        # 实例化SAGEConve，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型                

        if self.do_mask:            
            self.mask = nn.Embedding(output_dim,2,_weight=torch.log(torch.rand(output_dim,2)))
            self.mask_c = mask_c
        elif do_omp:            
            self.dummy_mask = nn.Parameter(torch.zeros(output_dim,1)) # prevent torch complaining no parameter
            self.omp_mask = torch.zeros(output_dim,1) # use to store results            
            self.omp_reward = do_omp

    def forward(self, graph, inputs):
        # 输入是节点的特征
        with graph.local_scope():
            adj = graph.adj(scipy_fmt='csr')
            h = torch.as_tensor(ut.floyd_warshall(adj,directed=False), dtype=torch.float)

            if self.do_mask:
                mask_ = F.gumbel_softmax(self.mask.weight,hard=True)[:, :1].T
                h = h * mask_
        return h

    def mask_loss(self, output, epoch, edge_mask=None):
        """
        Handles node and edge-identity prediction
        param: edge_mask is indexed by (i,j) in row-major order
        param: output is masked output from GNN
        edge_features assumed to be one-hot vector encoded in same order
        """
        
        if edge_mask != None:
            # use masked nodes to predict all edge identities
            cross_prod = list(product(output, output))            
            output = torch.stack(list(map(lambda x: torch.cat(x), cross_prod)))
            output_detach = output.clone().detach()
            y = edge_mask.long()
            sample_mask = ut.sample_mask(y)
        else:
            # use masked nodes to predict all node identities
            output_detach = output.clone().detach()
            y = torch.arange(len(output_detach))
            sample_mask = (y >= 0)

        if self.do_omp:      
            if epoch == 1:      
                best_coef, best_intercept, omp_mask = classify(output_detach[sample_mask], y[sample_mask], False, self.omp_reward)
                self.omp_mask[:, 0] = torch.as_tensor(omp_mask)
            else:
                return 0 * self.dummy_mask.sum()
        else:
            best_coef, best_intercept = classify(output_detach[sample_mask], y[sample_mask])
            
            
        best_coef = torch.from_numpy(best_coef).detach().float()        
        best_intercept = torch.from_numpy(best_intercept).detach().float()        
        loss = nn.CrossEntropyLoss()(output @ best_coef.T + best_intercept, y)
        
        
        # return mask_loss
        # return self.mask_c * mask_loss
        # return loss 
        if self.do_omp:
            return loss + 0 * self.dummy_mask.sum()
        else:
            mask_loss = F.gumbel_softmax(self.mask.weight,hard=True)[:,:1].sum()
            return loss + self.mask_c * mask_loss 

    def decide_action_mask(self, output, batch_size):
        if self.do_omp:
            set_indicator = self.omp_mask.data
        else:
            set_indicator = F.gumbel_softmax(self.mask.weight,hard=True)[:,:1]
        set_vector = []
        for i in range(batch_size):
            temp_set = []
            for index, value in enumerate(set_indicator[:, i]):
                if value == 1.:
                    temp_set.append(index)
            set_vector.append(temp_set)
        return set_vector, set_indicator

    def loss(self, set_indicator, reward_vector, output, epoch):
        lp = self.lp_loss(output, set_indicator)
        loss = torch.FloatTensor(reward_vector) * lp
        return torch.mean(loss), torch.mean(lp)  # /(output.shape[0])

    def lp_loss(self, output, set_indicator):
        lp = torch.sum(set_indicator * torch.log(output + EPS) +
                       (1 - set_indicator) * (torch.log(1 - output + EPS)), dim=0)

        return -lp



# GCN layer base dgl
class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats,num_layers,
                 input_dim, hidden_dim, output_dim, num_hidden_layers=2, mask_c=.0,act='sigmoid'):
        super(GCN,self).__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.do_mask = mask_c > .0

        # 实例化SAGEConve，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型
        
        for i in range(1, num_hidden_layers+1):
            setattr(self, f"conv{i}", GraphConv(hid_feats if i > 1 else in_feats, 
                                                hid_feats if i < num_hidden_layers else out_feats, 
                                                norm='both', weight=True, bias=True, activation=None, allow_zero_in_degree=False))

        self.mlp = mlp.MLP(self.num_layers, self.input_dim, self.hidden_dim, self.output_dim, act=act)

        if self.do_mask:            
            self.mask = nn.Embedding(output_dim,2,_weight=torch.log(torch.rand(output_dim,2)))
            self.edge_mask = nn.Embedding(output_dim*output_dim,2,_weight=torch.log(torch.rand(output_dim*output_dim,2)))
            self.mask_c = mask_c

    def forward(self, graph, inputs):
        # 输入是节点的特征
        with graph.local_scope():
            h = inputs
            for i in range(1, self.num_hidden_layers+1):
                h = getattr(self, f"conv{i}")(graph, h)
                h = F.relu(h)
                h = F.dropout(h, 0.5, training=self.training)

            h = self.mlp(h)
        return h

    def mask_loss(self, output, epoch, edge_mask=None):
        """
        Handles node and edge-identity prediction
        param: edge_mask is indexed by (i,j) in row-major order
        param: output is masked output from GNN
        edge_features assumed to be one-hot vector encoded in same order
        """
            
        if edge_mask != None:
            # use masked nodes to predict all edge identities
            cross_prod = list(product(output, output))            
            output = torch.stack(list(map(lambda x: torch.cat(x), cross_prod)))
            output_detach = output.clone().detach()
            y = edge_mask.long()
            sample_mask = ut.sample_mask(y)
        else:
            # use masked nodes to predict all node identities
            output_detach = output.clone().detach()
            y = torch.arange(len(output_detach))
            sample_mask = (y >= 0)

        best_coef, best_intercept = classify(output_detach[sample_mask], y[sample_mask])
        best_coef = torch.from_numpy(best_coef).detach().float()        
        best_intercept = torch.from_numpy(best_intercept).detach().float()        
        loss = nn.CrossEntropyLoss()(output @ best_coef.T + best_intercept, y)
        mask_loss = F.gumbel_softmax(self.mask.weight,hard=True)[:,:1].sum()
        # return mask_loss
        # return self.mask_c * mask_loss
        # return loss 
        return loss + self.mask_c * mask_loss

    def mvc_mask_loss(self, output, epoch, edge_mask):
        """
        Edge-identity prediction
        param: edge_mask is indexed by (i,j) in row-major order
        param: output is masked output from GNN
        edge_features assumed to be one-hot vector encoded in same order
        """
            
        # use masked nodes to predict all edge identities
        cross_prod = list(product(output, output))            
        output = torch.stack(list(map(lambda x: torch.cat(x), cross_prod)))
        output_detach = output.clone().detach()
        y = edge_mask.long()
        sample_mask = ut.sample_mask(y)

        best_coef, best_intercept = classify(output_detach[sample_mask], y[sample_mask])
        best_coef = torch.from_numpy(best_coef).detach().float()        
        best_intercept = torch.from_numpy(best_intercept).detach().float()        
        loss = nn.CrossEntropyLoss()(output @ best_coef.T + best_intercept, y)
        mask_loss = F.gumbel_softmax(self.mask.weight,hard=True)[:,:1].sum()
        # return mask_loss
        # return self.mask_c * mask_loss
        # return loss 
        return loss + self.mask_c * mask_loss

    def dom_k_mask_loss(self, output, k):
        """        
        """
        output_detach = output.clone().detach()
        mask_discrete = F.gumbel_softmax(self.mask.weight,hard=True)[:,:1]
        probs = F.relu(output_detach) * mask_discrete.T
        loss = torch.norm(probs.sum(axis=-1)-1,p=2)
        mask_loss = mask_discrete.sum()
        return mask_loss
        # return self.mask_c * mask_loss
        # return loss 
        return loss + self.mask_c * mask_loss
    
    def steiner_mask_loss(self, output, adj_inds):
        """                
        adj_inds: indices of upper-triangular edges
        """
        output_detach = output.clone().detach()
        mask_discrete = F.gumbel_softmax(self.mask.weight,hard=True)[:,:1]
        probs = F.relu(output_detach) * mask_discrete.T
        loss = torch.norm(probs.sum(axis=-1)-1,p=2)
        mask_loss = mask_discrete.sum()
        return mask_loss
        # return self.mask_c * mask_loss
        # return loss 
        return loss + self.mask_c * mask_loss


    def decide_action(self, output, batch_size):
        set_vector = []
        temp1 = output.detach().numpy()
        set_indicator = []
        for i in range(batch_size):
            a = np.random.binomial(1, temp1[:, i]) # (100,)
            while np.all(a == 0) or np.sum(a) > temp1.shape[0] - 2:

                # a = np.random.binomial(1, temp1[:, i])

                a=np.random.randint(0,2, size=(temp1.shape[0],))


            set_indicator.append(a)
        set_indicator = torch.tensor(set_indicator).transpose(1, 0) # (100, 32)

        for i in range(set_indicator.shape[1]):
            temp_set = []
            for index, value in enumerate(set_indicator[:, i]):
                if value > 0:
                    temp_set.append(index)
            # temp_set shape : (100,)
            set_vector.append(temp_set)

        # set_vector (32,), indices of landmarks only

        return set_vector, set_indicator

    def decide_action_mask(self, output, batch_size, set_indicator=None):
        set_indicator = set_indicator if set_indicator != None else F.gumbel_softmax(self.mask.weight,hard=True)[:,:1]
        set_vector = []
        for i in range(batch_size):
            temp_set = []
            for index, value in enumerate(set_indicator[:, i]):
                if value == 1.:
                    temp_set.append(index)
            set_vector.append(temp_set)
        return set_vector, set_indicator

    def decide_edge_action_mask(self, output, batch_size, adj):
        _, set_indicator = self.decide_action_mask(output,batch_size,F.gumbel_softmax(self.edge_mask.weight,hard=True)[:,:1])
        n = adj.shape[0]
        assert batch_size == set_indicator.shape[-1]
        set_vector = []
        for i in range(batch_size):
            temp_vector = []
            for e in range(len(set_indicator[:, i])):
                if not adj[e//n,e%n]:
                    set_indicator[e][i] = 0
                if set_indicator[e][i]:
                    temp_vector.append(e)
            set_vector.append(temp_vector)
        return set_vector, set_indicator

    def loss(self, set_indicator, reward_vector, output, epoch):
        lp = self.lp_loss(output, set_indicator)
        loss = torch.FloatTensor(reward_vector) * lp
        return torch.mean(loss), torch.mean(lp)  # /(output.shape[0])

    def lp_loss(self, output, set_indicator):
        lp = torch.sum(set_indicator * torch.log(output + EPS) +
                       (1 - set_indicator) * (torch.log(1 - output + EPS)), dim=0)

        return -lp


class GAT(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats,num_heads,
                 num_layers, input_dim, hidden_dim, output_dim):
        super(GAT,self).__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 实例化SAGEConve，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型
        self.conv1 = GATConv(self.in_feats, self.hid_feats, self.num_heads, feat_drop=0.5, attn_drop=0.5,
                           negative_slope=0.0, residual=False,activation=None, allow_zero_in_degree=False, bias=True)
        self.conv2 = GATConv(self.hid_feats, self.out_feats, self.num_heads, feat_drop=0.5, attn_drop=0.5,
                           negative_slope=0.0, residual=False,activation=None, allow_zero_in_degree=False, bias=True)
        self.mlp = mlp.MLP(self.num_layers, self.input_dim, self.hidden_dim, self.output_dim)

    def forward(self, graph, inputs):
        # 输入是节点的特征
        with graph.local_scope():
            h = self.conv1(graph, inputs)
            h = torch.mean(h, dim=1)
            h = F.relu(h)
            h = F.dropout(h, 0.5, training=self.training)
            h = self.conv2(graph, h)
            h = F.relu(h)
            h = F.dropout(h, 0.5, training=self.training)
            h = torch.mean(h,dim=1)
            h = self.mlp(h)
        return h

    def decide_action(self, output, batch_size):
        set_vector = []
        temp1 = output.detach().numpy()
        set_indicator = []
        for i in range(batch_size):
            a = np.random.binomial(1, temp1[:, i])
            while np.all(a == 0) or np.sum(a) > temp1.shape[0] - 2:
                a = np.random.binomial(1, temp1[:, i])
            set_indicator.append(a)
        set_indicator = torch.tensor(set_indicator).transpose(1, 0)

        for i in range(set_indicator.shape[1]):
            temp_set = []
            for index, value in enumerate(set_indicator[:, i]):
                if value > 0:
                    temp_set.append(index)
            set_vector.append(temp_set)

        return set_vector, set_indicator

    def loss(self, set_indicator, reward_vector, output, epoch):
        lp = self.lp_loss(output, set_indicator)
        loss = torch.FloatTensor(reward_vector) * lp
        return torch.mean(loss), torch.mean(lp)  # /(output.shape[0])

    def lp_loss(self, output, set_indicator):
        lp = torch.sum(set_indicator * torch.log(output + EPS) +
                       (1 - set_indicator) * (torch.log(1 - output + EPS)), dim=0)

        return -lp


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats,aggregator_type,
                 num_layers, input_dim, hidden_dim, output_dim, num_hidden_layers=2, mask_c=.0):
        super(SAGE,self).__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.aggregator_type = aggregator_type
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.do_mask = mask_c > .0        

        for i in range(1, num_hidden_layers+1):
            setattr(self, f"sage{i}", SAGEConv(hid_feats if i > 1 else in_feats, 
                                                hid_feats if i < num_hidden_layers else out_feats, 
                                                self.aggregator_type, feat_drop=0.5, bias=True, norm=None,activation=None))
       
        self.mlp = mlp.MLP(self.num_layers, self.input_dim, self.hidden_dim, self.output_dim)

        if self.do_mask:            
            self.mask = nn.Embedding(output_dim,2,_weight=torch.log(torch.rand(output_dim,2)))
            self.mask_c = mask_c        

    def forward(self, graph, inputs):
        # 输入是节点的特征
        with graph.local_scope():
            h = inputs
            for i in range(1, self.num_hidden_layers+1):
                h = getattr(self, f"sage{i}")(graph, h)
                h = F.relu(h)
                h = F.dropout(h, 0.5, training=self.training)

            h = self.mlp(h)
            if self.do_mask:
                mask_ = F.gumbel_softmax(self.mask.weight,hard=True)[:, :1].T
                h = h * mask_
        return h

    def mask_loss(self, output, epoch, edge_mask=None):
        """
        Handles node and edge-identity prediction
        param: edge_mask is indexed by (i,j) in row-major order
        param: output is masked output from GNN
        edge_features assumed to be one-hot vector encoded in same order
        """
            
        if edge_mask != None:
            # use masked nodes to predict all edge identities
            cross_prod = list(product(output, output))            
            output = torch.stack(list(map(lambda x: torch.cat(x), cross_prod)))
            output_detach = output.clone().detach()
            y = edge_mask.long()
            sample_mask = ut.sample_mask(y)
        else:
            # use masked nodes to predict all node identities
            output_detach = output.clone().detach()
            y = torch.arange(len(output_detach))
            sample_mask = (y >= 0)

        best_coef, best_intercept = classify(output_detach[sample_mask], y[sample_mask])
        best_coef = torch.from_numpy(best_coef).detach().float()        
        best_intercept = torch.from_numpy(best_intercept).detach().float()        
        loss = nn.CrossEntropyLoss()(output @ best_coef.T + best_intercept, y)
        mask_loss = F.gumbel_softmax(self.mask.weight,hard=True)[:,:1].sum()
        # return mask_loss
        # return self.mask_c * mask_loss
        # return loss 
        return loss + self.mask_c * mask_loss

    def decide_action(self, output, batch_size):
        set_vector = []
        temp1 = output.detach().numpy()
        set_indicator = []
        for i in range(batch_size):
            a = np.random.binomial(1, temp1[:, i])
            while np.all(a == 0):
                a = np.random.binomial(1, temp1[:, i])
            set_indicator.append(a)
        set_indicator = torch.tensor(set_indicator).transpose(1, 0)

        for i in range(set_indicator.shape[1]):
            temp_set = []
            for index, value in enumerate(set_indicator[:, i]):
                if value > 0:
                    temp_set.append(index)
            set_vector.append(temp_set)

        return set_vector, set_indicator

    def decide_action_mask(self, output, batch_size):
        set_indicator = F.gumbel_softmax(self.mask.weight,hard=True)[:,:1]
        set_vector = []
        for i in range(batch_size):
            temp_set = []
            for index, value in enumerate(set_indicator[:, i]):
                if value == 1.:
                    temp_set.append(index)
            set_vector.append(temp_set)
        return set_vector, set_indicator
  
    def loss(self, set_indicator, reward_vector, output, epoch):
        lp = self.lp_loss(output, set_indicator)
        loss = torch.FloatTensor(reward_vector) * lp
        return torch.mean(loss), torch.mean(lp)  # /(output.shape[0])

    def lp_loss(self, output, set_indicator):
        lp = torch.sum(set_indicator * torch.log(output + EPS) +
                       (1 - set_indicator) * (torch.log(1 - output + EPS)), dim=0)

        return -lp


class GIN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats,aggregator_type,
                 fun_num_layers,num_layers, input_dim, hidden_dim, output_dim, num_hidden_layers=2, mask_c=.0):
        super(GIN,self).__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats

        self.aggregator_type = aggregator_type
        self.fun_num_layers = fun_num_layers
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.do_mask = mask_c > .0

        for i in range(1, num_hidden_layers+1):
            setattr(self, f"mlp_fun{i}", mlp.MLP_fun(self.fun_num_layers, hid_feats if i > 1 else in_feats, self.hid_feats, hid_feats if i < num_hidden_layers else out_feats))            
            setattr(self, f"gin{i}", GINConv(getattr(self, f"mlp_fun{i}"), aggregator_type, init_eps=0, learn_eps=True))

        self.mlp = mlp.MLP(self.num_layers, self.input_dim, self.hidden_dim, self.output_dim)

        if self.do_mask:            
            self.mask = nn.Embedding(output_dim,2,_weight=torch.log(torch.rand(output_dim,2)))
            self.mask_c = mask_c

    def forward(self, graph, inputs):
        # 输入是节点的特征
        with graph.local_scope():
            h = inputs
            for i in range(1, self.num_hidden_layers+1):
                h = getattr(self, f"gin{i}")(graph, h)
                h = F.relu(h)
                h = F.dropout(h, 0.5, training=self.training)

            h = self.mlp(h)
            if self.do_mask:
                mask_ = F.gumbel_softmax(self.mask.weight,hard=True)[:, :1].T
                h = h * mask_

        return h

    def mask_loss(self, output, epoch, edge_mask=None):
        """
        Handles node and edge-identity prediction
        param: edge_mask is indexed by (i,j) in row-major order
        param: output is masked output from GNN
        edge_features assumed to be one-hot vector encoded in same order
        """
            
        if edge_mask != None:
            # use masked nodes to predict all edge identities
            cross_prod = list(product(output, output))            
            output = torch.stack(list(map(lambda x: torch.cat(x), cross_prod)))
            output_detach = output.clone().detach()
            y = edge_mask.long()
            sample_mask = ut.sample_mask(y)
        else:
            # use masked nodes to predict all node identities
            output_detach = output.clone().detach()
            y = torch.arange(len(output_detach))
            sample_mask = (y >= 0)

        best_coef, best_intercept = classify(output_detach[sample_mask], y[sample_mask])
        best_coef = torch.from_numpy(best_coef).detach().float()        
        best_intercept = torch.from_numpy(best_intercept).detach().float()        
        loss = nn.CrossEntropyLoss()(output @ best_coef.T + best_intercept, y)
        mask_loss = F.gumbel_softmax(self.mask.weight,hard=True)[:,:1].sum()
        # return mask_loss
        # return self.mask_c * mask_loss
        # return loss 
        return loss + self.mask_c * mask_loss

    def decide_action(self, output, batch_size):
        set_vector = []
        temp1 = output.detach().numpy()
        set_indicator = []
        for i in range(batch_size):
            a = np.random.binomial(1, temp1[:, i])
            while np.all(a == 0) or np.sum(a) > temp1.shape[0] - 2:
                a = np.random.binomial(1, temp1[:, i])
            set_indicator.append(a)
        set_indicator = torch.tensor(set_indicator).transpose(1, 0)

        for i in range(set_indicator.shape[1]):
            temp_set = []
            for index, value in enumerate(set_indicator[:, i]):
                if value > 0:
                    temp_set.append(index)
            set_vector.append(temp_set)

        return set_vector, set_indicator

    def decide_action_mask(self, output, batch_size):
        set_indicator = F.gumbel_softmax(self.mask.weight,hard=True)[:,:1]
        set_vector = []
        for i in range(batch_size):
            temp_set = []
            for index, value in enumerate(set_indicator[:, i]):
                if value == 1.:
                    temp_set.append(index)
            set_vector.append(temp_set)
        return set_vector, set_indicator
  
    def loss(self, set_indicator, reward_vector, output, epoch):
        lp = self.lp_loss(output, set_indicator)
        loss = torch.FloatTensor(reward_vector) * lp
        return torch.mean(loss), torch.mean(lp)  # /(output.shape[0])

    def lp_loss(self, output, set_indicator):
        lp = torch.sum(set_indicator * torch.log(output + EPS) +
                       (1 - set_indicator) * (torch.log(1 - output + EPS)), dim=0)

        return -lp

class GINE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats,
                 fun_num_layers,num_layers, input_dim, hidden_dim, output_dim, mask_c=.0):
        super(GINE,self).__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats

        self.fun_num_layers = fun_num_layers
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mlp_fun1 = mlp.MLP_fun(self.fun_num_layers, self.in_feats, self.hid_feats, self.hid_feats)
        self.mlp_fun2 = mlp.MLP_fun(self.fun_num_layers, self.hid_feats, self.hid_feats, self.out_feats)

        self.gine1 = GINEConv(self.mlp_fun1, init_eps=0, learn_eps=True)
        self.gine2 = GINEConv(self.mlp_fun2, init_eps=0, learn_eps=True)
        self.mlp = mlp.MLP(self.num_layers, self.input_dim, self.hidden_dim, self.output_dim)

        self.do_mask = mask_c != .0

        if self.do_mask:            
            self.mask = nn.Embedding(output_dim,2,_weight=torch.log(torch.rand(output_dim,2)))
            self.mask_c = mask_c

    def forward(self, graph, inputs, e_inputs):
        # 输入是节点的特征
        with graph.local_scope():
            pass
            # h = self.gine1(graph, inputs, e_inputs)
            # h = F.relu(h)
            # h = F.dropout(h, 0.5, training=self.training)
            # h = self.gine2(graph, h, e_inputs)
            # h = F.relu(h)
            # h = F.dropout(h, 0.5, training=self.training)
            # h = self.mlp(h)
            # if self.do_mask:
            #     mask_ = F.gumbel_softmax(self.mask.weight,hard=True)[:, :1].T
            #     h = h * mask_
        return inputs

    def decide_action(self, output, batch_size):
        set_vector = []
        temp1 = output.detach().numpy()
        set_indicator = []
        for i in range(batch_size):
            a = np.random.binomial(1, temp1[:, i])
            while np.all(a == 0) or np.sum(a) > temp1.shape[0] - 2:
                a = np.random.binomial(1, temp1[:, i])
            set_indicator.append(a)
        set_indicator = torch.tensor(set_indicator).transpose(1, 0)

        for i in range(set_indicator.shape[1]):
            temp_set = []
            for index, value in enumerate(set_indicator[:, i]):
                if value > 0:
                    temp_set.append(index)
            set_vector.append(temp_set)

        return set_vector, set_indicator

    def decide_action_mask(self, output, batch_size):
        set_indicator = F.gumbel_softmax(self.mask.weight,hard=True)[:,:1]
        set_vector = []
        for i in range(batch_size):
            temp_set = []
            for index, value in enumerate(set_indicator[:, i]):
                if value == 1.:
                    temp_set.append(index)
            set_vector.append(temp_set)
        return set_vector, set_indicator


    def loss(self, set_indicator, reward_vector, output, epoch):
        lp = self.lp_loss(output, set_indicator)
        loss = torch.FloatTensor(reward_vector) * lp
        return torch.mean(loss), torch.mean(lp)  # /(output.shape[0])

    def lp_loss(self, output, set_indicator):
        lp = torch.sum(set_indicator * torch.log(output + EPS) +
                       (1 - set_indicator) * (torch.log(1 - output + EPS)), dim=0)

        return -lp


class EDGE(nn.Module):
    def __init__(self, in_feats, hid_feats,out_feats,num_layers, input_dim, hidden_dim, output_dim, num_hidden_layers=2, mask_c=.0):
        super(EDGE,self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.hid_feats = hid_feats
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.do_mask = mask_c > .0        

        for i in range(1, num_hidden_layers+1):
            setattr(self, f"edge{i}", EdgeConv(hid_feats if i > 1 else in_feats, 
                                                hid_feats if i < num_hidden_layers else out_feats, 
                                                batch_norm=False, allow_zero_in_degree=False))

        self.mlp = mlp.MLP(self.num_layers, self.input_dim, self.hidden_dim, self.output_dim)
        if self.do_mask:            
            self.mask = nn.Embedding(output_dim,2,_weight=torch.log(torch.rand(output_dim,2)))
            self.mask_c = mask_c        

    def forward(self, graph, inputs):
        # 输入是节点的特征
        with graph.local_scope():
            h = inputs
            for i in range(1, self.num_hidden_layers+1):
                h = getattr(self, f"edge{i}")(graph, h)
                h = F.relu(h)
                h = F.dropout(h, 0.5, training=self.training)

            h = self.mlp(h)
            if self.do_mask:
                mask_ = F.gumbel_softmax(self.mask.weight,hard=True)[:, :1].T
                h = h * mask_
        return h

    def mask_loss(self, output, epoch, edge_mask=None):
        """
        Handles node and edge-identity prediction
        param: edge_mask is indexed by (i,j) in row-major order
        param: output is masked output from GNN
        edge_features assumed to be one-hot vector encoded in same order
        """
            
        if edge_mask != None:
            # use masked nodes to predict all edge identities
            cross_prod = list(product(output, output))            
            output = torch.stack(list(map(lambda x: torch.cat(x), cross_prod)))
            output_detach = output.clone().detach()
            y = edge_mask.long()
            sample_mask = ut.sample_mask(y)
        else:
            # use masked nodes to predict all node identities
            output_detach = output.clone().detach()
            y = torch.arange(len(output_detach))
            sample_mask = (y >= 0)

        best_coef, best_intercept = classify(output_detach[sample_mask], y[sample_mask])
        best_coef = torch.from_numpy(best_coef).detach().float()        
        best_intercept = torch.from_numpy(best_intercept).detach().float()        
        loss = nn.CrossEntropyLoss()(output @ best_coef.T + best_intercept, y)
        mask_loss = F.gumbel_softmax(self.mask.weight,hard=True)[:,:1].sum()
        # return mask_loss
        # return self.mask_c * mask_loss
        # return loss 
        return loss + self.mask_c * mask_loss


    def decide_action(self, output, batch_size):
        set_vector = []
        temp1 = output.detach().numpy()
        set_indicator = []
        for i in range(batch_size):
            a = np.random.binomial(1, temp1[:, i])
            while np.all(a == 0):
                a = np.random.binomial(1, temp1[:, i])
            set_indicator.append(a)
        set_indicator = torch.tensor(set_indicator).transpose(1, 0)

        for i in range(set_indicator.shape[1]):
            temp_set = []
            for index, value in enumerate(set_indicator[:, i]):
                if value > 0:
                    temp_set.append(index)
            set_vector.append(temp_set)

        return set_vector, set_indicator

    def decide_action_mask(self, output, batch_size):
        set_indicator = F.gumbel_softmax(self.mask.weight,hard=True)[:,:1]
        set_vector = []
        for i in range(batch_size):
            temp_set = []
            for index, value in enumerate(set_indicator[:, i]):
                if value == 1.:
                    temp_set.append(index)
            set_vector.append(temp_set)
        return set_vector, set_indicator


    def loss(self, set_indicator, reward_vector, output, epoch):
        lp = self.lp_loss(output, set_indicator)
        loss = torch.FloatTensor(reward_vector) * lp
        return torch.mean(loss), torch.mean(lp)  # /(output.shape[0])

    def lp_loss(self, output, set_indicator):
        lp = torch.sum(set_indicator * torch.log(output + EPS) +
                       (1 - set_indicator) * (torch.log(1 - output + EPS)), dim=0)

        return -lp


class TAG(nn.Module):
    def __init__(self, in_feats, hid_feats,out_feats,num_layers, input_dim, hidden_dim, output_dim, num_hidden_layers=2, mask_c=.0):
        super(TAG,self).__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.do_mask = mask_c > .0        

        for i in range(1, num_hidden_layers+1):
            setattr(self, f"tag{i}", TAGConv(hid_feats if i > 1 else in_feats, hid_feats if i < num_hidden_layers else out_feats, k=1, bias=True, activation=None))

        self.mlp = mlp.MLP(self.num_layers, self.input_dim, self.hidden_dim, self.output_dim)

        if self.do_mask:            
            self.mask = nn.Embedding(output_dim,2,_weight=torch.log(torch.rand(output_dim,2)))
            self.mask_c = mask_c

    def forward(self, graph, inputs):
        # 输入是节点的特征
        with graph.local_scope():
            h = inputs
            for i in range(1, self.num_hidden_layers+1):
                h = getattr(self, f"tag{i}")(graph, h)
                h = F.relu(h)
                h = F.dropout(h, 0.5, training=self.training)

            h = self.mlp(h)
            if self.do_mask:
                mask_ = F.gumbel_softmax(self.mask.weight,hard=True)[:, :1].T
                h = h * mask_
        return h

    def mask_loss(self, output, epoch, edge_mask=None):
        """
        Handles node and edge-identity prediction
        param: edge_mask is indexed by (i,j) in row-major order
        param: output is masked output from GNN
        edge_features assumed to be one-hot vector encoded in same order
        """
            
        if edge_mask != None:
            # use masked nodes to predict all edge identities
            cross_prod = list(product(output, output))            
            output = torch.stack(list(map(lambda x: torch.cat(x), cross_prod)))
            output_detach = output.clone().detach()
            y = edge_mask.long()
            sample_mask = ut.sample_mask(y)
        else:
            # use masked nodes to predict all node identities
            output_detach = output.clone().detach()
            y = torch.arange(len(output_detach))
            sample_mask = (y >= 0)

        best_coef, best_intercept = classify(output_detach[sample_mask], y[sample_mask])
        best_coef = torch.from_numpy(best_coef).detach().float()        
        best_intercept = torch.from_numpy(best_intercept).detach().float()        
        loss = nn.CrossEntropyLoss()(output @ best_coef.T + best_intercept, y)
        mask_loss = F.gumbel_softmax(self.mask.weight,hard=True)[:,:1].sum()
        # return mask_loss
        # return self.mask_c * mask_loss
        # return loss 
        return loss + self.mask_c * mask_loss


    def decide_action(self, output, batch_size):
        set_vector = []
        temp1 = output.detach().numpy()
        set_indicator = []
        for i in range(batch_size):
            a = np.random.binomial(1, temp1[:, i])
            while np.all(a == 0):
                a = np.random.binomial(1, temp1[:, i])
            set_indicator.append(a)
        set_indicator = torch.tensor(set_indicator).transpose(1, 0)

        for i in range(set_indicator.shape[1]):
            temp_set = []
            for index, value in enumerate(set_indicator[:, i]):
                if value > 0:
                    temp_set.append(index)
            set_vector.append(temp_set)

        return set_vector, set_indicator

    def decide_action_mask(self, output, batch_size):
        set_indicator = F.gumbel_softmax(self.mask.weight,hard=True)[:,:1]
        set_vector = []
        for i in range(batch_size):
            temp_set = []
            for index, value in enumerate(set_indicator[:, i]):
                if value == 1.:
                    temp_set.append(index)
            set_vector.append(temp_set)
        return set_vector, set_indicator

    def loss(self, set_indicator, reward_vector, output, epoch):
        lp = self.lp_loss(output, set_indicator)
        loss = torch.FloatTensor(reward_vector) * lp
        return torch.mean(loss), torch.mean(lp)  # /(output.shape[0])

    def lp_loss(self, output, set_indicator):
        lp = torch.sum(set_indicator * torch.log(output + EPS) +
                       (1 - set_indicator) * (torch.log(1 - output + EPS)), dim=0)

        return -lp


