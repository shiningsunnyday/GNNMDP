import os
import argparse
import gnn_solver as gs
from models import train_gcn as ts
import torch
import numpy as np
import json
from models import utils as ut

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=369, help='Random seed.')  # 456,13
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--Early_stop', type=int, default=101,
                    help='Early_stop.')
#0.5
parser.add_argument('--lr', type=float, default=0.01,
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
parser.add_argument('--algo', type=str, default='algo2', choices=['gnn-mdp','gnn-mvc','algo2'],help='gnn-mdp, gnn-mvc algorithm 2')
parser.add_argument('--mask_c', type=float, default=.5,help='tradeoff between loss and mask for gnn-mdp')
parser.add_argument('--num_iters', type=int, default=3,help='num iterations')
parser.add_argument('--flag', type=int, required=True, help='flag')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.algo == 'algo2':
    args.lr = 0.001
    args.batch_size = 32
    args.epochs = 500 # 100
    args.seed = 369
    args.hidden = 32
    args.noisy_size = 16
    args.nclass = 1
    args.num_iters = 3 # 10
    

# if solve MDP using gcn module then set mod = 'gcn' etc
# if solve MVC use edge-feature model, e.g. 'gine
mod = 'gine'  # gcn,gin,sage,edge,tag
method_dir = f'results_{args.algo}/'

train = True

import repair_method as rm


flag=args.flag

for a, (dataset, datapath, dataset1, dataset2) in enumerate(ut.load_datapath(flag)):    

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
        else:
            raise

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

    torch.save(global_state, model_save_path_file)
    # 每代运行结果

    record_path= "/{}_{}_record.txt".format(dataset, mod)
    f = open(pathname + record_path, 'w+')
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
    f.write('args ={}\n'.format(json.dumps(args.__dict__)))

    f.close()
    print("Resolving set sample finished!")
    
    print("Begin to compute the dim!")
    iter = 150 # len(global_resolving_set)
    if len(global_resolving_set) ==0:
        global_resolving_set = list(range(len(ntable)))
    r_set, dim = rm.repair_iter(args, global_resolving_set, ntable, iter)
    record_repair_path = "/{}_{}_record_opt.txt".format(dataset, mod)
    f = open(pathname + record_repair_path, 'w+')
    f.write('global_dim ={}\n'.format(dim))
    f.write('resolving_set ={}\n'.format(str(r_set)))
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











