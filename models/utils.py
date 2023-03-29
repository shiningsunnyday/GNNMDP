import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
import json
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from tqdm import tqdm
from multiprocessing import Pool
import re
import os

NAMES = ['tree', 'gnm', 'gnp', 'cluster', 'rpg', 'watts']
FLAG_D = {'tree': [10,9,8,7,6,5,4,3,2,1], 
        'gnm': [250,300,352,410,450,520,605,650,700,800], 
        'gnp': [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], 
        'cluster': [5,11,15,20,30,40,51,62,75,80], 
        'rpg': [5,7,8,10,15,20,25,30,35,40], 
        'watts': [5,11,20,30,40,55,60,75,85,90]}

def add_diameter(dic, max_lim=10):
    for i, run in tqdm(enumerate(dic['runs'])):
        if 'num_hidden_layers' not in run:
            continue
        flag = run['flag']
        aa = run['a']
        for dataset, datapath, dataset1, dataset2, a in load_datapath(flag):
            if a == aa:
                break
        adj, _ = load_data_adj_ntable(datapath, dataset1, dataset2)
        dist_matrix = floyd_warshall(csr_matrix(adj),directed=False)
        if dist_matrix.max() > max_lim:
            continue
        run['diameter'] = dist_matrix.max()

    return dic

def load_lp(lp_path):
    data = json.load(open(lp_path))
    ret = defaultdict(lambda: defaultdict(dict))
    for dic in data['runs']:
        flag = dic['flag']
        a = dic['a']
        ret[flag][a] = dic
    return ret

def smooth(scalar, weight=0.85):    
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed

def fill_triangular(vec, dim, mode="lower"):
    """Fill an lower or upper triangular matrices with given vectors"""
#     num_examples, size = vec.shape
#     assert size == dim * (dim + 1) // 2
#     matrix = torch.zeros(num_examples, dim, dim).to(vec.device)
#     if mode == "lower":
#         idx = (torch.tril(torch.ones(dim, dim)) == 1)[None]
#     elif mode == "upper":
#         idx = (torch.triu(torch.ones(dim, dim)) == 1)[None]
#     else:
#         raise Exception("mode {} not recognized!".format(mode))
#     idx = idx.repeat(num_examples,1,1)
#     matrix[idx] = vec.contiguous().view(-1)
    num_examples, size = vec.shape
    assert size == dim * (dim + 1) // 2
    if mode == "lower":
        rows, cols = torch.tril_indices(dim, dim)
    elif mode == "upper":
        rows, cols = torch.triu_indices(dim, dim)
    else:
        raise Exception("mode {} not recognized!".format(mode))
    matrix = torch.zeros(num_examples, dim, dim).type(vec.dtype).to(vec.device)
    matrix[:, rows, cols] = vec
    return matrix

def encode_onehot(labels):
    classes = set(labels) # 打乱了顺序
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_steinerlib(filename, labels_path):
    num_nodes = []
    edges = []
    terminals = []
    with open(labels_path,'r') as f:
        data = [line.split(',') for line in f.readlines()]
        names = [line[0] for line in data]
        labels = [int(line[-1]) for line in data]
        a = filename.split('/')[-1].rstrip('.stp')
        ind = names.index(a)
        opt = labels[ind]
        
    with open(filename,'r') as f:
        data = [line.rstrip('\n') for line in f.readlines()]
        graph_start = data.index("SECTION Graph")+1
        graph_end = data.index("END", graph_start)
        terminals_start = data.index("SECTION Terminals")+1
        terminals_end = data.index("END", terminals_start)
        num_nodes = int(data[graph_start].split()[-1])
        num_edges = int(data[graph_start+1].split()[-1])
        num_terminals = int(data[terminals_start+1].split()[-1])
        edges = []
        terminals = []
        for i in range(graph_start+2, graph_end):
            _, v1, v2, w = data[i].split()
            edges.append([int(v1),int(v2),int(w)])
            edges.append([int(v2),int(v1),int(w)])
        edges = np.array(edges)
        
        for i in range(terminals_start+2, terminals_end):
            _, t = data[i].split()
            terminals.append(int(t))


        adj = sp.coo_matrix((edges[:,-1], (edges[:,0]-1,edges[:,1]-1)),
                        shape=(num_nodes,num_nodes),
                        dtype=np.float32)

    return adj, terminals, opt


def load_steiner(name):
    filename = f"./data/{name}/{name.lower()}.txt"
    res = []
    with open(filename, 'r') as f:
        while True:
            cur = f.readline()
            if not cur: break
            cur, _, _, _, _, opt = cur.split(',')
            datapath = f"./data/{name}/{cur}.stp"
            res.append((name,filename,datapath,datapath,cur))
    return res


def check(dist_matrix, n, N):
    res = set()
    for i in range(N):
        for j in range(i+1, N):
            if dist_matrix[n][i] == dist_matrix[n][j]:
                res.add(N*i+j)
    # inds = inds[np.argwhere(dist_row[inds[:,0]] == dist_row[inds[:,1]]).flatten()]
    # res += set((N*inds[:,0]+inds[:,1]).tolist())
    return res

def compute_ntable(adj):
    N = adj.shape[0]
    dist_matrix = floyd_warshall(csr_matrix(adj),directed=False)
    p = Pool(8)
    # inds = np.array([[i,j] for i in range(N) for j in range(N)])    
    # pargs = [(dist_matrix[i], inds, i, N) for i in range(N)]
    res = p.starmap(check, [(dist_matrix, i, N) for i in range(N)])
    # res = []
    # for parg in pargs:
    #     res.append(check(*parg))
    ntable = {}
    for i, rset in enumerate(res):
        ntable[i] = rset
    return ntable

def load_3sat(dataset1):
    adj = np.genfromtxt(dataset1,dtype=np.str_)
    n = int(re.search("adj_\d*",dataset1).group()[4:])
    N = int(re.search("n=\d*",dataset1).group()[2:])
    M = int(re.search("m=\d*",dataset1).group()[2:])
    edges = adj[:,0:2]
    weight = adj[:,-1]

    idx = [str(i + 1) for i in range(n)]
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges.flatten())),
                    dtype=np.int32).reshape(edges.shape)
    adj = sp.coo_matrix((weight, (edges[:, 0], edges[:, 1])),
                        shape=(n,n),
                        dtype=np.float32)       
    return n,N,M,adj

def load_sat():
    filename = f"./data/sat_to_mdp/3sat/uf20-91/"
    return [(None,None,filename+fname,None,None) for fname in os.listdir(filename)]
    

def load_datapath(flag):
    if flag==1:
        dataset = 'powerlawtree'
        d=[10,9,8,7,6,5,4,3,2,1]
    elif flag==2:
        dataset = 'gnm'
        d=[250,300,352,410,450,520,605,650,700,800]
    elif flag==3:
        dataset = 'gnp'
        d=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    elif flag == 4:
        dataset = 'cluster'
        d=[5,11,15,20,30,40,51,62,75,80]

    elif flag == 5:
        dataset = 'regular'
        d=[5,7,8,10,15,20,25,30,35,40]
    elif flag == 6:
        dataset = 'watts'
        d=[5,11,20,30,40,55,60,75,85,90]
    elif flag == 'uf20-91':
        dataset = 'uf20-91'
        d=[x.split('-')[-1].rstrip('.txt') for x in os.listdir("data/sat_to_mdp/3sat/uf20-91/")]
    ret = []
    for i in d:
        a = i
        if flag == 1:
            # powertree
            datapath = "data/random_powerlaw_tree/adj_natable/"
            dataset1 = "adj_100_{}.txt".format(a)
            dataset2 = "ntable_100_{}.txt".format(a)
        elif flag == 2:
            # gnm
            datapath = "data/random_gnm/adj_natable/"
            dataset1 = "adj_100_{}.txt".format(a)
            dataset2 = "ntable_100_{}.txt".format(a)
        elif flag == 3:
            # gnp
            datapath = "data/random_gnp/adj_natable/"
            dataset1 = "adj_100_{}.txt".format(a)
            dataset2 = "ntable_100_{}.txt".format(a)
        elif flag == 4:
            # cluster
            datapath = "data/random_cluster_graph/adj_natable/"
            dataset1 = "adj_100_{}_0.5.txt".format(a)
            dataset2 = "ntable_100_{}_0.5.txt".format(a)
        elif flag == 5:
            # regular
            datapath = "data/random_regular_graph/adj_natable/"
            dataset1 = "adj_100_{}.txt".format(a)
            dataset2 = "ntable_100_{}.txt".format(a)
        elif flag == 6:
            # watts
            datapath = "data/random_watts/adj_natable/"
            dataset1 = "adj_100_{}_0.5.txt".format(a)
            dataset2 = "ntable_100_{}_0.5.txt".format(a)
        elif flag == 'uf20-91':
            # 3sat
            datapath = "data/3sat/uf20-91/"
            dataset1 = "uf20-{}.txt".format(a)
            dataset2 = "uf20-{}_ntable.txt".format(a)            
        ret.append((dataset, datapath, dataset1, dataset2, a))
    return ret

def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_data_adj_ntable(path, dataset1,dataset2):
    """Load graph adjacent matrix and non-resolving table from txt files"""

    # To read the adjacent matrix and non-resolving table-- ntable-- from the txt file
    adj = np.genfromtxt("{}{}".format(path, dataset1),delimiter='',dtype=np.str_)
    ntable = np.genfromtxt("{}{}".format(path, dataset2), delimiter='/n', dtype=np.str_)

    # To generate the edges and their weights
    edges = adj[:,0:2]
    weight = adj[:,-1]

    # To generate index for eahc id number

    idx = [str(i + 1) for i in range(len(ntable))]

    idx_map = {j: i for i, j in enumerate(idx)}

    # To generate edge represented by id's index:0,1,2,....
    edges = np.array(list(map(idx_map.get, edges.flatten())),
                     dtype=np.int32).reshape(edges.shape)


    # To generate sparse adjacent matrix
    adj = sp.coo_matrix((weight, (edges[:, 0], edges[:, 1])),
                        shape=(len(ntable), len(ntable)),
                        dtype=np.float32)


    # # build symmetric adjacency matrix
    # non_normalize_sparse_adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # # Normalize ymmetric adjacency matrix
    # normalize_adj = normalize(non_normalize_sparse_adj +
    #                           sp.eye(non_normalize_sparse_adj.shape[0]))
    # # To torch tensor
    # normalize_sparse_adj = sparse_mx_to_torch_sparse_tensor(normalize_adj)

    # To generate ntable dict with key 0,1,2...., and  the non-resolving vertex-pairs as items
    ntable_dict ={}
    for i,j in enumerate(ntable):
        ntable_dict[i] =set(map(int,j.split(' ')))

    # idx_train = range(len(ntable))
    # idx_train = torch.LongTensor(idx_train)

    return adj,ntable_dict

def compute_etable(adj):
    # assumes graph is connected, with bidirectional edges
    assert adj.row.min() == 0
    n = adj.row.max() + 1

    def create_nr(adj):
        s = set()
        for (src, tgt) in zip(adj.row, adj.col):
            s.add((src, tgt))
        return s

    etable = defaultdict(lambda: create_nr(adj))
    m = 0
    for i, j in zip(adj.row, adj.col):
        etable[i] -= set([(i, j)])
        etable[j] -= set([(i, j)])
        m += 1
    return etable, m


def compute_ktable(adj,k):
    # assumes graph is connected, with bidirectional edges
    assert adj.row.min() == 0
    n = adj.row.max() + 1
    ktable = defaultdict(set)
    for i in range(n): ktable[i]
    dist_matrix = floyd_warshall(csr_matrix(adj),directed=False)
    for i in range(n):
        for j in range(n):
            if dist_matrix[i][j]>k:
                ktable[i].add(j)
                ktable[j].add(i)
    return ktable


def compute_ttable(adj,T):
    # assumes graph is connected, with bidirectional edges
    assert adj.row.min() == 0
    n = adj.row.max() + 1
    adj = adj.toarray()
    ttable = defaultdict(set)
    for i in range(n):
        for j in range(n):
            if adj[i][j] == 0: continue
            ttable[n*i+j] = set(T)
            if i in ttable[n*i+j]:
                ttable[n*i+j].remove(i)
            if j in ttable[n*i+j]:
                ttable[n*i+j].remove(j)

    return ttable


def compute_edge_mask(G):
    n = G.num_nodes()
    mask = torch.zeros(n*n) # for directed pairs of vertices
    vp_ids = set()
    src, tgt = G.edges()
    
    for (i, j) in zip(src, tgt):
        e = (n*i + j).item()
        mask[e] = 1
        vp_ids.add(e)

    vp_dict = dict(zip(sorted(list(vp_ids)), range(len(vp_ids))))
    m = G.num_edges()
    

    e_feats = torch.zeros(m)
    feats = torch.zeros(n,m)

    for (i, j) in zip(src, tgt):
        e = vp_dict[(n*i + j).item()]
        feats[i][e] = 1
        feats[j][e] = 1
        e_feats[e] = 1
        
    return mask, feats, e_feats

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
# def sample_resolving_set(output):
#     preds = output.max(1)[1].type_as(labels)
#     correct = preds.eq(labels).double()
#     correct = correct.sum()
#     return correct / len(labels)

def reward(set_vector, ntable):
    PANEL = (len(ntable))    
    reward_vector = []
    penal_vector = []
    for i in range(len(set_vector)):
        temp_panel = set(range((PANEL*(PANEL-1))//2))
        
        for index, value in enumerate(set_vector[i]):
            temp_panel = temp_panel.intersection(ntable[value])

        # # fun4
        M = len(temp_panel)
        # if len(set_vector[i])> len(ntable)-2 or M != 0:
        temp_reward = 1. / (len(set_vector[i]) + PANEL * M)
        # else:
        #     temp_reward = 1. / (len(set_vector[i]))


        reward_vector.append(temp_reward)
        penal_vector.append(list(temp_panel)) # for serializing
    local_max_reward = max(reward_vector)
    local_best_ind = set_vector[reward_vector.index(local_max_reward)]
    local_best_panel = penal_vector[reward_vector.index(local_max_reward)]

    return reward_vector, \
            penal_vector, \
            local_best_ind, \
            local_best_panel, \
            local_max_reward

def compute_mvc_reward(set_vector, etable):
    PANEL = (len(etable))    
    reward_vector = []
    penal_vector = []
    for i in range(len(set_vector)):
        temp_panel = []
        for index, value in enumerate(set_vector[i]):
            if index == 0:
                temp_panel = etable[value]
            else:
                temp_panel = temp_panel.intersection(etable[value])

        # # fun4
        M = len(temp_panel)
        # if len(set_vector[i])> len(ntable)-2 or M != 0:
        print(f"{M} unresolved edges")
        temp_reward = -M
        # temp_reward = 1. / (len(set_vector[i]) + PANEL * M)
        # else:
        #     temp_reward = 1. / (len(set_vector[i]))


        reward_vector.append(temp_reward)
        penal_vector.append(temp_panel)
    local_max_reward = max(reward_vector)
    local_best_ind = set_vector[reward_vector.index(local_max_reward)]
    local_best_panel = penal_vector[reward_vector.index(local_max_reward)]

    return reward_vector, \
            penal_vector, \
            local_best_ind, \
            local_best_panel, \
            local_max_reward

def compute_bisect_reward(set_vector, etable):
    PANEL = (len(etable))    
    reward_vector = []
    penal_vector = []
    for i in range(len(set_vector)):
        temp_panel = []
        for index, value in enumerate(set_vector[i]):
            if index == 0:
                temp_panel = etable[value]
            else:
                temp_panel = temp_panel.intersection(etable[value])

        # # fun4
        M = len(temp_panel)
        # if len(set_vector[i])> len(ntable)-2 or M != 0:
        print(f"{M} unresolved edges")
        temp_reward = -M
        # temp_reward = 1. / (len(set_vector[i]) + PANEL * M)
        # else:
        #     temp_reward = 1. / (len(set_vector[i]))


        reward_vector.append(temp_reward)
        penal_vector.append(temp_panel)
    local_max_reward = max(reward_vector)
    local_best_ind = set_vector[reward_vector.index(local_max_reward)]
    local_best_panel = penal_vector[reward_vector.index(local_max_reward)]

    return reward_vector, \
            penal_vector, \
            local_best_ind, \
            local_best_panel, \
            local_max_reward


def compute_dom_k_reward(set_vector, ktable):
    PANEL = (len(ktable))    
    reward_vector = []
    penal_vector = []
    for i in range(len(set_vector)):
        temp_panel = set(range(PANEL))
        
        for index, value in enumerate(set_vector[i]):
            temp_panel = temp_panel.intersection(ktable[value])

        # # fun4
        M = len(temp_panel)
        # if len(set_vector[i])> len(ntable)-2 or M != 0:
        temp_reward = 1. / (len(set_vector[i]) + PANEL * M)
        # else:
        #     temp_reward = 1. / (len(set_vector[i]))


        reward_vector.append(temp_reward)
        penal_vector.append(list(temp_panel)) # for serializing
    local_max_reward = max(reward_vector)
    local_best_ind = set_vector[reward_vector.index(local_max_reward)]
    local_best_panel = penal_vector[reward_vector.index(local_max_reward)]

    return reward_vector, \
            penal_vector, \
            local_best_ind, \
            local_best_panel, \
            local_max_reward

def compute_steiner_reward(set_vector, ttable, adj):
    PANEL = adj.shape[0]
    reward_vector = []
    penal_vector = []
    for i in range(len(set_vector)):
        temp_panel = set(range(PANEL))
        
        for index, value in enumerate(set_vector[i]):
            temp_panel = temp_panel.intersection(ttable[value])

        # # fun4
        M = len(temp_panel)
        # if len(set_vector[i])> len(ntable)-2 or M != 0:
        W = 0.
        for e in set_vector[i]:
            i, j = e//PANEL, e%PANEL
            W += adj[i,j]
        temp_reward = -float("inf") if M else -W
        # else:
        #     temp_reward = 1. / (len(set_vector[i]))


        reward_vector.append(temp_reward)
        penal_vector.append(list(temp_panel)) # for serializing
    local_max_reward = max(reward_vector)
    local_best_ind = set_vector[reward_vector.index(local_max_reward)]
    local_best_panel = penal_vector[reward_vector.index(local_max_reward)]

    return reward_vector, \
            penal_vector, \
            local_best_ind, \
            local_best_panel, \
            local_max_reward


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sample1(batch_size, node_num, noisy_size):
    fea = torch.Tensor(batch_size, node_num, noisy_size).uniform_(0.5)
  

    return fea

def sample2(batch_size, node_num, noisy_size):
    s = torch.ones(batch_size, node_num, noisy_size)
    return s


def sample_mask(y):
    """
    Generate a boolean mask for positives and equivalent amount of random negatives
    """
    pos_mask = (y == 1)
    num_pos = pos_mask.sum()
    ratio = (int)(len(y)/num_pos)
    neg_mask = np.arange(len(y)) % ratio == 0
    return pos_mask | neg_mask
    
