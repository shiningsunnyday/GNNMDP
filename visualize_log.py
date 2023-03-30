import matplotlib.pyplot as plt
import matplotlib
import json
from collections import defaultdict
from scipy import interpolate
import numpy as np
from models import utils as ut
from scipy.linalg import lstsq
from itertools import chain
from decimal import Decimal
import pandas as pd

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
        y = ut.smooth(rewards)
        plt.plot(x, y, label=f'{hparam}={key}')
        plt.scatter([np.argmax(rewards)], [max(rewards)], label=max(rewards))
        
    plt.legend()
    plt.yscale('log')
    return fig

def mismatch(run_dic, fixed={}):
    b = False
    for k, v in fixed.items():
        if k not in run_dic:
            b = True
            continue
        if isinstance(v, list):
            if run_dic[k] not in v:
                b = True
        elif run_dic[k] != v: 
            b = True
    return b


def vary_hparam(hparam, fixed={}, pool=lambda x:max(x), pool_args=[]):
    """
    fixed: dic of hparams to fix
    hparam: key of hparam to vary
    pool: takes in list of run_dics for hparam value, outputs number
    pool_args: args to pass to pool callabe, default empty
    plot avg(reward) over datasets
    """
    res = defaultdict(list)
    for run_dic in dic['runs']:
        if hparam not in run_dic:
            continue
        if mismatch(run_dic, fixed):
            continue

        rewards = run_dic['global_reward_list']
        if len(rewards) != run_dic['epochs']: 
            continue
        
        res[run_dic[hparam]] += [run_dic]

    fig = plt.figure()        
    res_keys = sorted(list(res.keys()))    

    for k in res_keys:
        print(f"pooling over {len(res[k])} runs for {hparam}={k}")
        
    res_v = [pool(res[k], *pool_args) for k in res_keys]
    res_keys = list(map(str, res_keys))
    plt.bar(res_keys, res_v)
    for i, v in enumerate(res_v):
        plt.text(i-0.4, v, "%.3f" % round(v,3), color='blue', fontweight='bold')
    print(res_keys, res_v)
    return fig

def fill_table(dic, hparam, fixed={}, pool=lambda x:max(x), pool_args=[]):
    """
    make df filled with results similar to vary_hparam
    """
    res = defaultdict(list)
    for run_dic in dic['runs']:
        if hparam:
            if hparam not in run_dic:
                continue
            if mismatch(run_dic, fixed):
                continue

            rewards = run_dic['global_reward_list']
            if len(rewards) != run_dic['epochs']: 
                continue
        
        res[run_dic[hparam] if hparam else ""] += [run_dic]

    res_keys = sorted(list(res.keys()))    

    for k in res_keys:
        print(f"pooling over {len(res[k])} runs for {hparam}={k}")

    res_v = {k: pool(res[k], *pool_args) for k in res_keys}
    data = {}
    rows = [[np.nan for _ in range(len(ut.FLAG_D[name]))] for name in ut.NAMES]
    for k, best_k in res_v.items():        
        for flag in best_k:
            for a, v in best_k[flag].items():
                name = ut.NAMES[flag-1]
                a_ind = ut.FLAG_D[name].index(a)
                rows[flag-1][a_ind] = v
        data[k] = list(chain(*rows))

    index = [f"{name}{ind}" for name in ut.NAMES for ind in range(1, len(ut.FLAG_D[name])+1)]
    df = pd.DataFrame(data, np.array(index))
    
    return df    

def pool_func(x, metric='reward', raw=False, report='reward'):
    """
    metric: use this to select the best run per (flag, a)
    report: report this for the return value (may be different)
    """
    pool = max
    if metric == 'reward':
        metric = lambda dic: max(dic['global_reward_list'])
    elif metric == 'lp_reward':
        metric = lambda dic: lp_dic[dic['flag']][dic['a']]['reward']
    elif metric == 'lp_ratio':
        metric = lambda dic: lp_dic[dic['flag']][dic['a']]['reward']/max(dic['global_reward_list'])
        pool = min
    else:
        raise

    if report == 'reward':
        report = lambda dic: max(dic['global_reward_list'])
    elif report == 'lp_reward':
        report = lambda dic: lp_dic[dic['flag']][dic['a']]['reward']
    elif report == 'lp_ratio':
        report = lambda dic: lp_dic[dic['flag']][dic['a']]['reward']/max(dic['global_reward_list'])
        pool = min
    elif report == 'metric dimension':
        report = lambda dic: dic['md'] if 'md' in dic else (dic['global_dim_opt'] if dic['global_dim'] else 0)
    elif report == 'nr':        
        report = lambda dic: int((1/max(dic['global_reward_list']) - dic['global_dim'])/100)
        
    else:
        raise

    best = defaultdict(lambda: defaultdict(lambda: (0., 0.)))
    for dic in x:
        a = dic['a']
        flag = dic['flag']

        cand = pool(best[flag][a][0], metric(dic))
        if cand != best[flag][a][0]: # since pool can be max or min
            best[flag][a] = (cand, report(dic))
      

    if raw:
        for flag in best:
            for a in best[flag]:
                best[flag][a] = best[flag][a][1]
        return best
    
    sum = 0
    for flag in best:
        inner_sum = 0
        for a in best[flag]:
            inner_sum += best[flag][a][1]
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
        if mismatch(run_dic, fixed):
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

    ax.view_init(azim=225, elev=50)

    ax.scatter(x,y,z)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    ax.set_zlim(*zlim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    
    return fig, ax


def fill_main_table(): 
    def join(df1, df2):
        for c1, c2 in zip(df1.columns, df2.columns):
            assert c1 == c2
            df1[c1] = df1[c1].combine(df2[c2], lambda x, y: (x, y))
        return df1

    dic = json.load(open('./results_gnn-mdp/new_logs-final.json'))
    fixed_dic = {
        "lr": 0.01, 
        'algo': 'gnn-mdp',
        'mask_c': 3,
        'epochs': 1000,
        'num_iters': 1,
    }
    df = fill_table(dic, "gnn_model", fixed_dic, pool=pool_func, pool_args=('reward', True, 'reward'))
    df = join(df, fill_table(dic, "gnn_model", fixed_dic, pool=pool_func, pool_args=('reward', True, 'metric dimension')))
    fixed_dic = {
        "lr": 0.003, 
        'algo': 'algo2',
    }
    dic = json.load(open('./results_algo2/new_logs-final.json'))
    df_algo2 = fill_table(dic, "gnn_model", fixed_dic, pool=pool_func, pool_args=('reward', True, 'reward'))
    df_algo2 = join(df_algo2, fill_table(dic, "gnn_model", fixed_dic, pool=pool_func, pool_args=('reward', True, 'metric dimension')))
    # df_ratio = df/df_algo2    
    # df_ratio = df_ratio.applymap(lambda x: '%.2E' % Decimal(x))
    # df_algo2 = df_algo2.applymap(lambda x: '%.2E' % Decimal(x))
    # df = df.applymap(lambda x: '%.2E' % Decimal(x))
    # df_ratio.to_csv('./gnns-final-gnnmdp-to-algo2.csv')    
    dic = json.load(open('./results_lp/lp_logs.json'))
    df_lp = fill_table(dic, "", {}, pool=pool_func, pool_args=('lp_reward', True, 'lp_reward'))    
    df_lp = join(df_lp, fill_table(dic, "", {}, pool=pool_func, pool_args=('lp_reward', True, 'metric dimension')))
    fixed_dic = {
        "lr": 0.003, 
        'algo': 'algo2',
        'mask_c': 3.0,
    }
    dic = json.load(open('./results_gnn-mdp/new_logsdistmask.json'))
    df_distmask = fill_table(dic, "", fixed_dic, pool=pool_func, pool_args=('reward', True, 'reward'))
    df_distmask = join(df_distmask, fill_table(dic, "", fixed_dic, pool=pool_func, pool_args=('reward', True, 'metric dimension')))
    dic = json.load(open('./results_gnn-mdp/new_logsdistmask_omp.json'))
    fixed_dic = {
        "epochs": 2, 
        'algo': 'gnn-mdp',
        'distmask_c': 1.0,
        'do_omp': True,
        'num_iters': 1,
    }
    df_omp = fill_table(dic, "", fixed_dic, pool=pool_func, pool_args=('reward', True, 'reward'))   
    df_omp = join(df_omp, fill_table(dic, "", fixed_dic, pool=pool_func, pool_args=('reward', True, 'metric dimension')))
    df_final = pd.DataFrame()
    for c in df:
        df_final[f"Mask{c.upper()}"] = df[c]
    for c in df_algo2:
        df_final[f"Hybrid{c.upper()}"] = df_algo2[c]    
    df_final[f"DistMask-Linear"] = df_distmask.values
    df_final[f"DistMask-OMP"] = df_omp.values
    df_final[f"MDP-LP"] = df_lp.values
    df_summary = df_final.groupby(by=lambda x:x.rstrip('1234567890'), sort=False)
    df_summary = df_summary.aggregate(lambda x: x.apply(pd.Series).mean(axis=0).aggregate(lambda x: tuple(x))).T
    
    sci_notation = lambda df: df.applymap(lambda x: ('%.2E' % Decimal(x[0]), int(x[1]) if x[1]==x[1] else x[1]))
    sci_notation(df).to_csv('./gnns-final.csv')
    sci_notation(df_algo2).to_csv('./gnns-final-algo2.csv')
    sci_notation(df_lp).to_csv('./lp-final.csv')
    sci_notation(df_distmask).to_csv('./distmask-final.csv')
    sci_notation(df_omp).to_csv('./omp-final.csv')   
    sci_notation(df_summary).to_csv('./mdp-summary.csv')
    sci_notation(df_final).to_csv('./mdp-final.csv')   

# path = './results_algo2/new_logs.json'
# path = './results_gnn-mdp/new_logs-layer-only.json' # hidden layer exp only
# path = './results_gnn-mdp/new_logs_and_layer.json' # new_logs.json + hidden layer exp
# path = './results_gnn-mdp/new_logs-final.json' # gnn architectures with default settings
# path = './results_gnn-mdp/new_logsdistmask.json' # distmask baseline
path = './results_gnn-mdp/new_logsdistmask_omp.json' # distmask omp baseline
lp_path = './results_lp/lp_logs.json'

lp_dic = ut.load_lp(lp_path)
f = open(path)
dic = json.load(f)
f.close()

font = {'size'   : 13}

matplotlib.rc('font', **font)

# fig = vary_hparam("mask_c", {
#     "lr": 0.01, 
#     'algo': 'gnn-mdp',
#     'epochs': 1000,
#     'flag': [2,3,4,5,6],
#     'num_iters': 1,
#     'gnn_model': 'gcn',
# }, pool=pool_func)
# plt.title("MaskGNN: Average reward vs Masking strength")
# plt.xlabel("Masking strength lambda")
# plt.ylabel("Reward R_G(S)")
# fig.savefig("./mask_c_new.png")
# fig.clear()
# plt.clf()

# fig = vary_hparam("lr", {
#     'algo': 'algo2',
#     'epochs': 100,
#     'num_iters': 10,
#     'gnn_model': 'gcn',
#     'flag': [2,3,4,5,6]
# }, pool=pool_func, pool_args=('reward',))
# plt.title("HybridGNN: Average reward vs Learning rate")
# plt.xlabel("learning rate")
# plt.ylabel("Reward R_G(S)")
# fig.savefig("./lr_algo2.png")



# fig = vary_hparam("mask_c", {
#     'algo': 'gnn-mdp',
#     'gnn_model': 'distmask',
# }, pool=pool_func)
# plt.title("DistMask: Average reward vs Masking strength")
# plt.xlabel("Masking strength lambda")
# plt.ylabel("Reward R_G(S)")
# plt.tick_params(axis='both', which='major', labelsize=10)
# plt.tick_params(axis='both', which='minor', labelsize=10)
# fig.savefig("./distmask_lambda.png")
# fig.clear()
# plt.clf()

# fig = vary_hparam("distmask_c", {
#     "epochs": 2, 
#     'algo': 'gnn-mdp',
#     'do_omp': True,
#     'num_iters': 1,
#     'flag': [2,3,4,5,6]
# }, pool=pool_func)
# plt.title("DistMask-OMP: Average reward vs Masking strength")
# plt.xlabel("Masking strength lambda_OMP")
# plt.ylabel("Reward R_G(S)")
# plt.tick_params(axis='both', which='major', labelsize=10)
# plt.tick_params(axis='both', which='minor', labelsize=10)
# fig.savefig("./distmask_lambda_omp.png")
# fig.clear()
# plt.clf()

# fig = vary_hparam("num_hidden_layers", {
#     "lr": 0.01, 
#     'algo': 'gnn-mdp',
#     'mask_c': 3,
#     'epochs': 1000,
#     'num_iters': 1,
#     'flag': [2,3,4,5,6],
#     'gnn_model': 'gcn',
# }, pool=pool_func)
# plt.title("MaskGCN: Average Reward vs Number of GCN layers")
# plt.xlabel("Number of GCN layers (K)")
# plt.ylabel("Reward R_G(S)")
# fig.savefig("./num_hidden_layers.png")
# fig.clear()
# plt.clf()

# fig = vary_hparam("gnn_model", {
#     "lr": 0.01, 
#     'algo': 'gnn-mdp',
#     'mask_c': 3,
#     'epochs': 1000,
#     'num_iters': 1,
# }, pool=pool_func, pool_args=(True,))
# plt.title("average reward vs gnn architecture")
# plt.xlabel("layer architecture")
# plt.ylabel("Reward R_G(S)")
# fig.savefig("./gnns-final.png")
# fig.clear()
# plt.clf()

# dic = ut.add_diameter(dic)

# fig, ax = heatmap("diameter", "num_hidden_layers", fixed={
#     'lr': 0.01,
#     'algo': 'gnn-mdp',
#     'mask_c': 3,
#     'flag': [2,3,4,5,6],
#     'epochs': 1000,
#     'num_iters': 1,
#     'gnn_model': 'gcn',
# }, pool=pool_heatmap, xlim=[0,10], ylim=[0,16], zlim=[0,0.14], plot_surface=False)
# plt.title("MaskGCN runs scatterplot of (reward, diameter, layers)")
# ax.set_xlabel('Diameter')
# ax.set_ylabel('Number of GCN layers (K)')
# ax.set_zlabel('Reward (R_G(S))')
# fig.savefig(f"./nhl_diam_3d_f=2-6.png")
# fig.clear()
# plt.clf()

# fig = vary_hparam("lr", {
#     'algo': 'gnn-mdp',
#     'mask_c': 3.0,
#     'num_hidden_layers': 2,
#     'epochs': 1000,
#     'flag': [2,3,4,5,6],
#     'num_iters': 1,
#     'gnn_model': 'gcn',
# }, pool=pool_func)
# plt.title("MaskGNN: Average reward vs Learning rate")
# plt.xlabel("Learning rate")
# plt.ylabel("Reward R_G(S)")
# fig.savefig("./gnn-mdp_lr.png")
# fig.clear()

# fig = plot_runs('mask_c', fixed={
#     "lr": 0.01, 
#     'algo': 'gnn-mdp',
#     'epochs': 1000,
#     'num_iters': 1,
#     'gnn_model': 'gcn',
# fig.savefig("./interpolate_runs.png")

if __name__ == '__main__':
    fill_main_table()
    pass