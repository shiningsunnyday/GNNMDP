import matplotlib.pyplot as plt
import json
from collections import defaultdict
from scipy import interpolate
import numpy as np

path = './results_gnn-mdp/logs.json'
f = open(path)
dic = json.load(f)
f.close()

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
    plot avg(reward) over datasets
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
    plot avg(reward) over datasets
    """
    res = defaultdict(list)
    for run_dic in dic['runs']:
        for k, v in fixed.items():
            if run_dic[k] != v: 
                continue      

        rewards = run_dic['global_reward_list']
        if len(rewards) != run_dic['epochs']: 
            continue
        
        res[run_dic[hparam]] += [max(rewards)]

    fig = plt.figure()        
    res_keys = list(res.keys())
    res_v = [pool(res[k]) for k in res_keys]
    str_labels = list(map(str, res_keys))
    plt.bar(str_labels, res_v)
    return fig

# fig = vary_hparam("mask_c", {
#     "lr": 0.01, 
#     'algo': 'gnn-mdp',
#     'epochs': 1000,
#     'num_iters': 1
# })
# fig.savefig("./mask_c.png")

fig = plot_runs('mask_c', fixed={
    "lr": 0.01, 
    'algo': 'gnn-mdp',
    'epochs': 500,
    'num_iters': 1,
    'gnn_model': 'gcn',
    'flag': 5
})
fig.savefig("./interpolate_runs.png")