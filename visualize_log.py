import matplotlib.pyplot as plt
import json
from collections import defaultdict
from scipy import interpolate
import numpy as np
from models import utils as ut
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from scipy.linalg import lstsq
from tqdm import tqdm

def smooth(scalar, weight=0.85):    
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed

def plot_runs(hparam, fixed={}):
    """
    fixed: dic of hparams to fix
    hparam: key of hparam to vary
    plot (hparam value, avg(reward))
    """
    res = {}
    best_res = defaultdict(int)
    for run_dic in dic['runs']:
        rewards = run_dic['global_reward_list']
        if len(rewards) != run_dic['epochs']: 
            continue

        b = True
        for k, v in fixed.items():
            if type(run_dic[k]) != type(v):
                breakpoint()
            if run_dic[k] != v: 
                b = False      

        if not b: 
            continue

        key = run_dic[hparam]  
        cand = max(run_dic['global_reward_list'])
        if cand > best_res[key]: 
            best_res[key] = cand
            res[key] = run_dic
    
    fig = plt.figure()
    for key, run_dic in res.items():
        rewards = run_dic['global_reward_list']
        x = range(run_dic['epochs'])
        y = smooth(rewards)
        plt.plot(x, y, label=f'{hparam}={key}')
        plt.scatter([np.argmax(rewards)], [max(rewards)], label=max(rewards))
        
    plt.legend()
    plt.yscale('log')
    return fig

def vary_hparam(hparam, fixed={}, pool=lambda x:max(x)):
    """
    fixed: dic of hparams to fix
    hparam: key of hparam to vary
    pool: takes in list of run_dics for hparam value, outputs number
    plot avg(reward) over datasets
    """
    res = defaultdict(list)
    for run_dic in dic['runs']:
        if hparam not in run_dic:
            continue
        for k, v in fixed.items():
            if k not in run_dic:
                continue
            if run_dic[k] != v: 
                continue      

        rewards = run_dic['global_reward_list']
        if len(rewards) != run_dic['epochs']: 
            continue
        
        res[run_dic[hparam]] += [run_dic]

    fig = plt.figure()        
    res_keys = sorted(list(res.keys()))    

    for k in res_keys:
        print(f"pooling over {len(res[k])} runs for {hparam}={k}")
        
    res_v = [pool(res[k]) for k in res_keys]
    res_keys = list(map(str, res_keys))
    plt.bar(res_keys, res_v)
    for i, v in enumerate(res_v):
     plt.text(i-0.25, v, str(round(v,4)), color='blue', fontweight='bold')
    print(res_keys, res_v)
    return fig

def pool_func(x):
    best = defaultdict(lambda: defaultdict(lambda: 0.))
    for dic in x:
        a = dic['a']
        flag = dic['flag']
        best[flag][a] = max(best[flag][a], max(dic['global_reward_list']))
    
    sum = 0
    for flag in best:
        inner_sum = 0
        for a in best[flag]:
            inner_sum += best[flag][a]
        sum += inner_sum/len(best[flag])
    return sum/len(best)

def pool_heatmap(x):
    return sum(max(run['global_reward_list']) for run in x)/len(x)

def heatmap(x1, x2, xlim=[], ylim=[], zlim=[], fixed={}, pool=None, plot_surface=False):
    res = defaultdict(list)
    for run_dic in dic['runs']:
        if x1 not in run_dic:
            continue
        if x2 not in run_dic:
            continue
        for k, v in fixed.items():
            if k not in run_dic:
                continue
            if run_dic[k] != v: 
                continue      

        rewards = run_dic['global_reward_list']
        if len(rewards) != run_dic['epochs']: 
            continue
        
        res[(run_dic[x1],run_dic[x2])] += [run_dic]    

    x, y, z = [], [], []
    for k, v in res.items():
        x.append(k[0])
        y.append(k[1])
        z.append(pool(v))

    data = np.array([x,y,z]).T
    X, Y = np.meshgrid(np.arange(xlim[0],xlim[1],0.5),np.arange(ylim[0],ylim[1],0.5))
    XX = X.flatten()
    YY = Y.flatten()
    A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
    C,_,_,_ = lstsq(A, data[:,2])
    
    # evaluate it on a grid
    Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    if plot_surface:
        ax.plot_surface(X,Y,Z)

    ax.view_init(azim=225)
    ax.scatter(x,y,z)
    ax.set_zlim(*zlim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    
    
    return fig


path = './results_gnn-mdp/new_logs-layer.json'
f = open(path)
dic = json.load(f)
f.close()


    

# fig = vary_hparam("mask_c", {
#     "lr": 0.01, 
#     'algo': 'gnn-mdp',
#     'epochs': 1000,
#     'num_iters': 1,
#     'gnn_model': 'gcn',
# }, pool=pool_func)
# plt.title("average reward vs masking strength")
# plt.xlabel("lambda")
# plt.ylabel("1/(|S|+|V|*|NR(S))")
# fig.savefig("./mask_c_new.png")
# fig.clear()

# fig = vary_hparam("lr", {
#     'algo': 'algo2',
#     'epochs': 100,
#     'num_iters': 10,
#     'gnn_model': 'gcn'
# }, pool=pool_func)
# plt.title("average reward vs learning rate")
# plt.xlabel("learning rate")
# plt.ylabel("1/(|S|+|V|*|NR(S))")
# fig.savefig("./lr_algo2.png")
# fig.clear()

# fig = vary_hparam("num_hidden_layers", {
#     "lr": 0.01, 
#     'algo': 'gnn-mndp',
#     'mask_c': 3,
#     'epochs': 1000,
#     'num_iters': 1,
#     'gnn_model': 'gcn',
# }, pool=pool_func)
# plt.title("average reward vs num conv layers")
# plt.xlabel("num conv layers")
# plt.ylabel("1/(|S|+|V|*|NR(S))")
# fig.savefig("./num_hidden_layers.png")
# fig.clear()

# def add_diameter(dic, max_lim=10):
#     for i, run in tqdm(enumerate(dic['runs'])):
#         if 'num_hidden_layers' not in run:
#             continue
#         flag = run['flag']
#         aa = run['a']
#         for dataset, datapath, dataset1, dataset2, a in ut.load_datapath(flag):
#             if a == aa:
#                 break
#         adj, _ = ut.load_data_adj_ntable(datapath, dataset1, dataset2)
#         dist_matrix = floyd_warshall(csr_matrix(adj),directed=False)
#         if dist_matrix.max() > max_lim:
#             continue
#         run['diameter'] = dist_matrix.max()

#     return dic

# dic = add_diameter(dic)
# fig = heatmap("diameter", "num_hidden_layers", fixed={
#     'lr': 0.01,
#     'algo': 'gnn-mdp',
#     'mask_c': 3,
#     'epochs': 1000,
#     'num_iters': 1,
#     'gnn_model': 'gcn',
# }, pool=pool_heatmap, xlim=[0,10], ylim=[0,16], zlim=[0,0.10])
# fig.savefig("./nhl_diam_3d.png")
# fig.clear()

fig = vary_hparam("lr", {
    'algo': 'gnn-mdp',
    'mask_c': 3.0,
    'num_hidden_layers': 2,
    'epochs': 1000,
    'num_iters': 1,
    'gnn_model': 'gcn',
}, pool=pool_func)
plt.title("average reward vs learning rate")
plt.xlabel("learning rate")
plt.ylabel("1/(|S|+|V|*|NR(S))")
fig.savefig("./gnn-mdp_lr.png")
fig.clear()

# fig = plot_runs('mask_c', fixed={
#     "lr": 0.01, 
#     'algo': 'gnn-mdp',
#     'epochs': 1000,
#     'num_iters': 1,
#     'gnn_model': 'gcn',
# })
# fig.savefig("./interpolate_runs.png")