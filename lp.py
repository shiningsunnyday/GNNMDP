from mip import Model, xsum, minimize, BINARY, OptimizationStatus
import numpy as np
import scipy.sparse as sp
from models import utils as ut
import argparse 
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
import re
from collections import defaultdict
import json
import time
import os
    
parser = argparse.ArgumentParser()

parser.add_argument('--flag', type=int, required=False, help='flag')
parser.add_argument('--task', type=str, required=True, choices=['mvc', 'mdp', 'dom_k', 'steiner', '3sat'])
parser.add_argument('--k', type=int, required=False)

args = parser.parse_args()
flag = args.flag
log_path = f'results_{args.task}/lp_logs.json'
if os.path.exists(log_path):
    f = open(log_path, 'r')
    data = json.load(f)
else:
    data = {'runs':[]}

if args.task == 'steiner':
    load_iterable = ut.load_steiner(flag)
else:
    load_iterable = ut.load_datapath(flag)

for i, (dataset, path, dataset1, dataset2, a) in enumerate(load_iterable):

    dic = {}
    dic['a'] = a
    if args.task == 'steiner':
        adj, _, dic['opt'] = ut.load_steinerlib(flag, path, a)
        dic['name'] = flag
    elif args.task == '3sat':
        n, N, M, adj = ut.load_3sat(dataset1) 
    else:
        adj, ntable = ut.load_data_adj_ntable(path, dataset1, dataset2)   
        dic['flag'] = flag
    n = adj.shape[0]
    E = zip(adj.row, adj.col)    
    
    if args.task == 'mvc':                 
        m = Model("mvc")
        x = [m.add_var(var_type=BINARY) for i in range(n)]
        m.objective = minimize(xsum(x[i] for i in range(n)))

        for (i, j) in E:
            m += xsum((x[i], x[j])) >= 1

        status = m.optimize(max_seconds=7500)
        selected = [i for i in range(n) if x[i].x >= 0.99]
        print("{}/{} selected nodes: {}".format(len(selected), n, selected))
    elif args.task == 'dom_k':
        dist_matrix = floyd_warshall(csr_matrix(adj),directed=False)
        k = args.k
        m = Model(args.task)
        x = [m.add_var(var_type=BINARY) for i in range(n)]
        m.objective = minimize(xsum(x[i] for i in range(n)))
        neighbors = defaultdict(list)
        for i in range(n):
            for j in range(n):
                if dist_matrix[i][j] <= k:
                    neighbors[i].append(j)
        for i in range(n):
            m += xsum(x[j] for j in neighbors[i]) >= 1
        start = time.time()
        status = m.optimize(max_seconds=7500)
        dic['selected'] = [i for i in range(n) if x[i].x >= 0.99]
        dic['time'] = time.time() - start
        dic['k'] = k
        print(status)
    elif args.task == 'steiner':
        pass
    elif args.task == '3sat':
        dic['beta'] = N+M
    else:
        dist_matrix = floyd_warshall(csr_matrix(adj),directed=False)
        m = Model("mdp")
        x = [m.add_var(var_type=BINARY) for i in range(n)]
        m.objective = minimize(xsum(x[i] for i in range(n)))
        for i in range(n):
            for j in range(i+1, n):
                m += xsum(abs(dist_matrix[i][k]-dist_matrix[j][k]) * x[k] for k in range(n)) >= 1
        start = time.time()
        status = m.optimize(max_seconds=7500)
        time_took = time.time() - start
        selected = [i for i in range(n) if x[i].x >= 0.99]
        temp_panel = set(range((n*(n-1))//2))
        for i in selected: 
            temp_panel = temp_panel.intersection(ntable[i])

        print("{}/{} selected nodes: {}".format(len(selected), n, selected))
        print("reward:", 1/(len(selected) + len(temp_panel)*n))                   
               
        dic['selected'] = selected
        dic['reward'] = 1/(len(selected) + len(temp_panel)*n)
        dic['time']= time_took

    if args.task not in ['steiner','3sat']:
        if status == OptimizationStatus.OPTIMAL:
            print('optimal solution cost {} found'.format(m.objective_value))
        elif status == OptimizationStatus.FEASIBLE:
            print('sol.cost {} found, best possible: {}'.format(m.objective_value, m.objective_bound))
        elif status == OptimizationStatus.NO_SOLUTION_FOUND:
            print('no feasible solution found, lower bound is: {}'.format(m.objective_bound))
    else:
        print(dic)

    f = open(log_path, 'w+')
    data['runs'].append(dic)
    json.dump(data, f)     
    f.close()
