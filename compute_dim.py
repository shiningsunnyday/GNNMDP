import os
import argparse
import gnn_solver as gs
from models import train_gcn as ts
import torch
import numpy as np
from models import utils as ut

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
parser.add_argument('--nclass', type=int, default=1,
                    help='Number of output units.')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Number of batch size for training.')
parser.add_argument('--noisy_size', type=int, default=16,
                    help='Number of batch size for training.')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)


# if solve MDP using gcn module then set mod = 'gcn' etc
mod = 'gin'  # gcn,gin,sage,edge,tag

train = True

import repair_method as rm


flag=2
if flag==1:
    dataset = 'powerlawtree'
    d=[1] #[10,9,8,7,6,5,4,3,2,1]
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

    # Save path folder for running results
    pathname = 'results/{}/{}/'.format(dataset,a)
    tmep_path = os.path.exists(pathname)
    if not tmep_path:
        os.makedirs(pathname)

    model_save_path = 'results/{}/{}/model_weights'.format(dataset,a)
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
    for iter in range(10):

        print('Iteration: ', iter+1)

        # features = ut.sample1(args.batch_size, 100, args.noisy_size)
        # print(features)
        if train:
            local_ave_time, local_reward_list, local_loss_list, local_best_ind, \
            local_best_ind_panel_set, local_max_reward, local_best_state,ntable \
                = gs.gnn_solver(args,mod,datapath,dataset1,dataset2,ts,
                                train=True,model_path=None)
        else:
            local_ave_time, local_reward_list, local_loss_list, local_best_ind, \
            local_best_ind_panel_set, local_max_reward,ntable\
                = gs.gnn_solver(args, mod, datapath, dataset1, dataset2, ts,
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
    f = open(pathname + record_path, 'a+')
    f.write('global_dim ={}\n'.format(global_dim))
    f.write('global_pan ={}\n'.format(global_pan))

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











