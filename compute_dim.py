import os
import argparse
import gnn_solver as gs
from models import train_gcn as ts
import torch
import numpy as np
import json
from models import utils as ut
import matplotlib.pyplot as plt
from copy import deepcopy

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=369, help='Random seed.')  # 456,13
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--Early_stop', type=int, default=101,
                    help='Early_stop.')
#0.5
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--nclass', type=int, default=100,
                    help='Number of output units.')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Number of batch size for training.')
parser.add_argument('--noisy_size', type=int, default=16,
                    help='Number of batch size for training.')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--algo', type=str, default='algo2', choices=['gnn-mdp','gnn-mvc','gnn-dom_k','gnn-bisect','gnn-steiner','gnn-3sat','algo2'],help='gnn-mdp, gnn-mvc algorithm 2')
parser.add_argument('--dom_k',type=int,default=10,help='k for k-distance dominating set')
parser.add_argument('--mask_c', type=float, default=.5,help='tradeoff between loss and mask for gnn-mdp')
parser.add_argument('--distmask_c', type=float, default=.5,help='tradeoff between score and k for do_omp baseline')
parser.add_argument('--do_omp', type=bool, default=False,help='whether to do orthogonal matching pursuit instead of mask')
parser.add_argument('--num_iters', type=int, default=3,help='num iterations')
parser.add_argument('--gnn_model', type=str, default='gcn', help='gnn model', choices=['distmask', 'gcn','gin','sage','edge','tag','gine'])
parser.add_argument('--num_hidden_layers', type=int, default=2, help='number of conv layers')
parser.add_argument('--flag', required=False, default='', help='flag (int) or dataset name for steiner trees')
parser.add_argument('--suffix',type=str,default='',help='if non-empty, add suffix to log file to differentiate')

args = parser.parse_args()
if args.flag.isnumeric():
    args.flag = int(args.flag)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.algo == 'algo2':
    # args.lr = 0.001
    args.batch_size = 32
    args.epochs = 100
    args.seed = 369
    args.hidden = 32
    args.noisy_size = 16
    args.nclass = 1
    args.num_iters = 10
    

# if solve MDP using gcn module then set mod = 'gcn' etc
# if solve MVC use edge-feature model, e.g. 'gine
mod = args.gnn_model  # gcn,gin,sage,edge,tag,gine for mvc
method_dir = 'results_{}/'.format(args.algo)

train = True

import repair_method as rm

def did_run(log_path, arg_dict):
    if not os.path.exists(log_path): return False
    cur_data = json.load(open(log_path))['runs']
    for run in cur_data:
        is_run = True
        for k, v in arg_dict.items():
            if k not in run or run[k] != v:
                is_run = False
        if is_run:
            return True
    return False

flag=args.flag

if args.algo == 'gnn-steiner':
    load_iterable = ut.load_steiner(args.flag)
else:
    load_iterable = ut.load_datapath(flag)

for _, (dataset, datapath, dataset1, dataset2, a) in enumerate(load_iterable):
    
    arg_dict = args.__dict__
    arg_dict['a'] = a 

    log_path = f'results_{args.algo}/new_logs{args.suffix}.json'

    if did_run(log_path, arg_dict): 
        print("ran before, skipping")
        continue

    # Save path folder for running results
    pathname = method_dir + '{}/{}/'.format(dataset,a)
    tmep_path = os.path.exists(pathname)
    if not tmep_path:
        os.makedirs(pathname)

    model_save_path = method_dir + '{}/{}/model_weights'.format(dataset,a)
    tmep_path = os.path.exists(model_save_path)
    if not tmep_path:
        os.makedirs(model_save_path)

    model_save_file = '\parameter_{}.pkl'.format(mod)
    model_save_path_file = model_save_path + model_save_file

    global_dim =0
    global_pan = 0
    global_loss_list = []
    global_reward_list =[]
    global_reward = -1000
    global_time = 0
    global_resolving_set = []
    global_panel_set = []
    global_state = 0
    dim_list = []
    time_list = []
    res_list = []
    pan_list = []
    best_seed = 0
    ntable = 0

    #'''
    # solve MDP by algorithm2
    for iter in range(args.num_iters):

        print('Iteration: ', iter+1)

        # features = ut.sample1(args.batch_size, 100, args.noisy_size)
        # print(features)
        if args.algo=='algo2':
            algo = gs.gnn_solver
        elif args.algo=='gnn-mdp':
            algo = gs.gnn_mdp
        elif args.algo=='gnn-mvc':
            algo = gs.gnn_mvc
        elif args.algo=='gnn-bisect':
            algo = gs.gnn_bisect
        elif args.algo=='gnn-steiner':
            algo = gs.gnn_steiner
        elif args.algo=='gnn-dom_k':
            algo = gs.gnn_dom_k
        elif args.algo=='gnn-3sat':
            algo = gs.gnn_mdp

        if train:
            local_ave_time, local_reward_list, local_loss_list, local_best_ind, \
            local_best_ind_panel_set, local_max_reward, local_best_state,ntable \
                = algo(args,mod,datapath,dataset1,dataset2,ts,
                                train=True,model_path=None)
        else:
            local_ave_time, local_reward_list, local_loss_list, local_best_ind, \
            local_best_ind_panel_set, local_max_reward,ntable\
                = algo(args, mod, datapath, dataset1, dataset2, ts,
                                train=False, model_path=model_save_path_file)

        if global_reward < local_max_reward:
            global_reward = local_max_reward
            global_loss_list = local_loss_list
            global_reward_list = local_reward_list
            global_resolving_set = local_best_ind
            global_panel_set = local_best_ind_panel_set
            global_time = local_ave_time
            if train:
                global_state = local_best_state
            best_seed = args.seed
        global_dim = len(global_resolving_set)
        global_pan = len(global_panel_set)
        dim_list.append(len(global_resolving_set))
        res_list.append(global_resolving_set)
        pan_list.append(global_panel_set)
        time_list.append(global_time)

        plt.figure()
        fig_output_filename1 = "/reward_{}_{}_{}.pdf".format(dataset, a, iter)
        plt.plot(local_reward_list, label="Reward")
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        plt.title('Episode mean reward curve')
        plt.savefig(pathname + fig_output_filename1)
        plt.close()

    torch.save(global_state, model_save_path_file)
    # 每代运行结果

    record_path= "/{}_{}_record.txt".format(dataset, mod)
    f = open(pathname + record_path, 'a+')
    f.write('global_dim ={}\n'.format(global_dim))
    f.write('global_pan ={}\n'.format(global_pan))
    f.write('global_best_reward ={}\n'.
         format(max(global_reward_list)))

    f.write('metric_dimension_list ={}\n'.
         format(str(dim_list)))
    f.write('metric_resolving_set_list ={}\n'.
         format(str(res_list)))
    f.write('panel_set_list ={}\n'.
         format(str(pan_list)))
    f.write('time_list ={}\n'.
         format(str(time_list)))
    f.write('global_rewards_list ={}\n'.
         format(str(global_reward_list)))
    f.write('global_loss_list ={}\n'.
         format(str(global_loss_list)))
    f.write('best_seed ={}\n'.
         format(str(best_seed)))
    
    try:

        f.write('args ={}\n'.format(json.dumps(arg_dict)))
    except:
        breakpoint()

    f.close()
    print("Resolving set sample finished!")
    
    print("Begin to compute the dim!")
    iter = 150 # len(global_resolving_set)
    if len(global_resolving_set) ==0:
        global_resolving_set = list(range(len(ntable)))
    r_set, dim = rm.repair_iter(args, global_resolving_set, ntable, iter)
    record_repair_path = "/{}_{}_record_opt.txt".format(dataset, mod)
    f = open(pathname + record_repair_path, 'a+')
    f.write('global_dim ={}\n'.format(dim))
    f.write('resolving_set ={}\n'.format(str(r_set)))
    f.close()    

    print('global_dim ={}\n'.format(dim))
    print('resolving_set ={}\n'.format(str(r_set)))
    
    if os.path.exists(log_path):
        f = open(log_path, 'r')
        data = json.load(f)
    else:
        f = open(log_path, 'w+')
        data = {'runs':[]}
    
    f.close()
    dic = deepcopy(arg_dict)
    dic['global_dim'] = global_dim
    dic['global_pan'] = global_pan
    dic['global_reward_list'] = global_reward_list
    dic['metric_dimension_list'] = dim_list
    dic['metric_resolving_set_list'] = res_list
    dic['panel_set_list'] = pan_list
    dic['global_loss_list'] = global_loss_list
    dic['global_dim_opt'] = dim
    dic['resolving_set_opt'] = r_set

    f = open(log_path, 'w+')
    data['runs'].append(dic)
    try:
        json.dump(data, f)
    except:
        breakpoint()
    f.close()

    #'''

    # /////////////////////////////////////////////// solve MDP by greed only if needed

    '''
    #  solve MDP by greed only
    print('solving MDP with greedy repair policy')
    iter2 = 250  # 200000 #len(global_resolving_set) #  greed
    _, ntable = ut.load_data_adj_ntable(datapath, dataset1, dataset2)
    global_resolving_set = list(range(len(ntable)))
    r_set, dim = rm.repair_iter(args, global_resolving_set, ntable, iter2)
    record_repair_path = "/{}_{}_record_repair_only.txt".format(dataset, mod)
    f = open(pathname + record_repair_path, 'a+')
    f.write('global_dim ={}\n'.format(dim))
    f.write('resolving_set ={}\n'.format(str(r_set)))
    f.close()
    
    '''











