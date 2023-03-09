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

parser.add_argument('--fname', type=str, default='./LP_algo/results/records/lp_logs.txt')

args = parser.parse_args()
log_path = f'./results_lp/lp_logs.json'

names = ['tree', 'gnm', 'gnp', 'cluster', 'rpg', 'watts']
runs = {n: [{} for i in range(10)] for n in names}

with open(args.fname) as f:
    for cur in f:        
        name, rset = cur.split('=')
        flagname, ind = name.split('_')
        runs[flagname][int(ind)-1] = rset
        
res = {"runs": []}
for flag in [1, 2, 3, 4, 5, 6]:
    for i, (dataset, path, dataset1, dataset2, a) in enumerate(ut.load_datapath(flag)):
        if not runs[names[flag-1]][i]: 
            continue

        rset = runs[names[flag-1]][i].split(',')
        rset = list(map(int, rset))
        md = sum(rset)
        
        # print("parsed",flag,"index",i,"a",a)
        adj, ntable = ut.load_data_adj_ntable(path, dataset1, dataset2)   
        n = len(ntable)

        temp_panel = set(range((n*(n-1))//2))
        for node, i in enumerate(rset): 
            if i == 1:
                temp_panel = temp_panel.intersection(ntable[node])

        dic = {'reward': 1/(md + len(temp_panel)*n), 'md': md, 'flag': flag, 'a': a}
        res['runs'].append(dic)
        print(dic['reward'])


json.dump(res, open(log_path, 'w+'))
        