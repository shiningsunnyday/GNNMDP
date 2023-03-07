from mip import Model, xsum, minimize, BINARY
import numpy as np
from models import utils as ut
import argparse 
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
import json
import time
import os

parser = argparse.ArgumentParser()

parser.add_argument('--flag', type=int, required=True, help='flag')
parser.add_argument('--task', type=str, required=True, choices=['mvc', 'mdp'])

args = parser.parse_args()
flag = args.flag
log_path = f'results_lp/lp_logs.json'

for i, (dataset, path, dataset1, dataset2, a) in enumerate(ut.load_datapath(flag)):
    adj, ntable = ut.load_data_adj_ntable(path, dataset1, dataset2)   
    n = len(ntable)
    E = zip(adj.row, adj.col)
    if args.task == 'mvc':                 
        m = Model("mvc")
        x = [m.add_var(var_type=BINARY) for i in range(n)]
        m.objective = minimize(xsum(x[i] for i in range(n)))

        for (i, j) in E:
            m += xsum((x[i], x[j])) >= 1

        m.optimize()
        selected = [i for i in range(n) if x[i].x >= 0.99]
        print("{}/{} selected nodes: {}".format(len(selected), n, selected))
    else:
        dist_matrix = floyd_warshall(csr_matrix(adj),directed=False)
        m = Model("mdp")
        x = [m.add_var(var_type=BINARY) for i in range(n)]
        m.objective = minimize(xsum(x[i] for i in range(n)))
        for i in range(n):
            for j in range(i+1, n):
                m += xsum(abs(dist_matrix[i][k]-dist_matrix[j][k]) * x[k] for k in range(n)) >= 1
        start = time.time()
        m.optimize()
        time_took = time.time() - start
        selected = [i for i in range(n) if x[i].x >= 0.99]
        temp_panel = set(range((n*(n-1))//2))
        for i in selected: 
            temp_panel = temp_panel.intersection(ntable[i])

        print("{}/{} selected nodes: {}".format(len(selected), n, selected))
        print("reward:", 1/(len(selected) + len(temp_panel)*n))        
        
        if os.path.exists(log_path):
            f = open(log_path, 'r')
            data = json.load(f)
        else:
            f = open(log_path, 'w+')
            data = {'runs':[]}
        
        f.close()
        
        dic = {}
        dic['a'] = a
        dic['flag'] = flag
        dic['selected'] = selected
        dic['reward'] = 1/(len(selected) + len(temp_panel)*n)
        dic['time']= time_took

        f = open(log_path, 'w+')
        data['runs'].append(dic)
        json.dump(data, f)     
        f.close()
