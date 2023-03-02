# modules
import torch
import numpy as np
from itertools import product
import torch.nn as nn
# The wn is a fun of normalization for param weight matrix
from torch.nn.utils import weight_norm as wn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv,GATConv,SAGEConv,GINConv,EdgeConv,TAGConv
from sklearn.linear_model import LogisticRegression
from models import mlp
EPS = 1e-30

def classify(feats):
    clf=LogisticRegression().fit(feats,np.arange(len(feats)))
    return clf.coef_ # (n classes, n features)

# GCN layer base dgl
class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats,num_layers,
                 input_dim, hidden_dim, output_dim, mask_c=.0):
        super(GCN,self).__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.do_mask = mask_c > .0

        # 实例化SAGEConve，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型
        self.conv1 = GraphConv(self.in_feats, self.hid_feats, norm='both', weight=True, bias=True,
                                     activation=None, allow_zero_in_degree=False)
        self.conv2 = GraphConv(self.hid_feats, out_feats, norm='both', weight=True, bias=True,
                                     activation=None, allow_zero_in_degree=False)
        self.mlp = mlp.MLP(self.num_layers, self.input_dim, self.hidden_dim, self.output_dim)

        if self.do_mask:            
            self.mask = nn.Embedding(output_dim,2,_weight=torch.log(torch.rand(output_dim,2)))
            self.mask_c = mask_c

    def forward(self, graph, inputs):
        # 输入是节点的特征
        with graph.local_scope():
            h = self.conv1(graph, inputs)
            h = F.relu(h)
            h = F.dropout(h, 0.5, training=self.training)
            h = self.conv2(graph, h)
            h = F.relu(h)
            h = F.dropout(h, 0.5, training=self.training)
            h = self.mlp(h)
            if self.do_mask:
                mask_ = F.gumbel_softmax(self.mask.weight,hard=True)[:, :1].T
                h = h * mask_
        return h

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

    def mask_loss(self, output, epoch, edge_mask=None):
        """
        Handles node and edge-identity prediction
        param: edge_mask is indexed by (i,j) in row-major order
        param: output is masked output from GNN
        edge_features assumed to be one-hot vector encoded in same order
        """
        
        output = output.clone().detach()
        if edge_mask:
            # use masked nodes to predict all edge identities            
            edge_output = torch.stack(map(lambda x,y: torch.cat((x,y))), product(output, output))
            output = edge_output[edge_mask]
        else:
            # use masked nodes to predict all node identities
            pass

        best_coef = torch.from_numpy(classify(output)).detach().float()
        loss = nn.CrossEntropyLoss()(output @ best_coef.T, torch.arange(len(output)))
        mask_loss = F.gumbel_softmax(self.mask.weight,hard=True)[:,:1].sum()

        # return mask_loss
        return loss + self.mask_c * mask_loss
    
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
                 num_layers, input_dim, hidden_dim, output_dim):
        super(SAGE,self).__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.aggregator_type = aggregator_type
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.sage1 = SAGEConv(self.in_feats, self.hid_feats, self.aggregator_type,
                              feat_drop=0.5, bias=True, norm=None,activation=None)
        self.sage2 = SAGEConv(self.hid_feats, self.out_feats, self.aggregator_type,
                              feat_drop=0.5, bias=True, norm=None,activation=None)
        self.mlp = mlp.MLP(self.num_layers, self.input_dim, self.hidden_dim, self.output_dim)

    def forward(self, graph, inputs):
        # 输入是节点的特征
        with graph.local_scope():
            h = self.sage1(graph, inputs)
            h = F.relu(h)
            h = F.dropout(h, 0.5, training=self.training)
            h = self.sage2(graph, h)
            h = F.relu(h)
            h = F.dropout(h, 0.5, training=self.training)
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


class GIN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats,aggregator_type,
                 fun_num_layers,num_layers, input_dim, hidden_dim, output_dim):
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
        self.mlp_fun1 = mlp.MLP_fun(self.fun_num_layers, self.in_feats, self.hid_feats, self.hid_feats)
        self.mlp_fun2 = mlp.MLP_fun(self.fun_num_layers, self.hid_feats, self.hid_feats, self.out_feats)

        self.gin1 = GINConv(self.mlp_fun1, aggregator_type, init_eps=0, learn_eps=True)
        self.gin2 = GINConv(self.mlp_fun2, aggregator_type, init_eps=0, learn_eps=True)
        self.mlp = mlp.MLP(self.num_layers, self.input_dim, self.hidden_dim, self.output_dim)

    def forward(self, graph, inputs):
        # 输入是节点的特征
        with graph.local_scope():
            h = self.gin1(graph, inputs)
            h = F.relu(h)
            h = F.dropout(h, 0.5, training=self.training)
            h = self.gin2(graph, h)
            h = F.relu(h)
            h = F.dropout(h, 0.5, training=self.training)
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


class EDGE(nn.Module):
    def __init__(self, in_feats, hid_feats,out_feats,num_layers, input_dim, hidden_dim, output_dim):
        super(EDGE,self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.hid_feats = hid_feats
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.edge1 = EdgeConv(self.in_feats, self.hid_feats, batch_norm=False, allow_zero_in_degree=False)
        self.edge2 = EdgeConv(self.hid_feats, self.out_feats, batch_norm=False, allow_zero_in_degree=False)
        self.mlp = mlp.MLP(self.num_layers, self.input_dim, self.hidden_dim, self.output_dim)

    def forward(self, graph, inputs):
        # 输入是节点的特征
        with graph.local_scope():
            h = self.edge1(graph, inputs)
            h = F.relu(h)
            h = F.dropout(h, 0.5, training=self.training)
            h = self.edge2(graph, h)
            h = F.relu(h)
            h = F.dropout(h, 0.5, training=self.training)
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


class TAG(nn.Module):
    def __init__(self, in_feats, hid_feats,out_feats,num_layers, input_dim, hidden_dim, output_dim):
        super(TAG,self).__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.tag1 = TAGConv(self.in_feats, self.hid_feats, k=1, bias=True, activation=None)
        self.tag2 = TAGConv(self.hid_feats, self.hid_feats, k=1, bias=True, activation=None)
        self.mlp = mlp.MLP(self.num_layers, self.input_dim, self.hidden_dim, self.output_dim)

    def forward(self, graph, inputs):
        # 输入是节点的特征
        with graph.local_scope():
            h = self.tag1(graph, inputs)
            h = F.relu(h)
            h = F.dropout(h, 0.5, training=self.training)
            h = self.tag2(graph, h)
            h = F.relu(h)
            h = F.dropout(h, 0.5, training=self.training)
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


