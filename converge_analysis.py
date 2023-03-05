import os
import argparse
import gnn_solver as gs
from models import train_gcn as ts
import torch
import numpy as np
from models import utils as ut
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=369, help='Random seed.')  # 456,13
parser.add_argument('--epochs', type=int, default=1000,
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
parser.add_argument('--algo', type=str, default='algo2', choices=['gnn-mdp','gnn-mvc','algo2'],help='gnn-mdp, gnn-mvc algorithm 2')


args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

mod = 'gcn'  # gcn,gat,sage,gin,edge,tag
train = True

import repair_method as rm

# data
flag=2
if flag==1:
    dataset = 'powerlawtree'
    d=[1] #[10,9,8,7,6,5,4,3,2,1]
elif flag==2:
    dataset = 'gnm'
    d=[250]#[250,300,352,410,450,520,605,650,700,800]
elif flag==3:
    dataset = 'gnp'
    d=[0.6,0.1,0.2,0.3,0.4,0.5,0.05,0.7,0.8,0.9]

elif flag == 4:
    dataset = 'cluster'
    d=[5,11,15,20,30,40,51,62,75,80]

elif flag == 5:
    dataset = 'regular'
    d=[5,7,8,10,15,20,25,30,35,40]
elif flag == 6:
    dataset = 'watts'
    d=[5,11,20,30,40,55,60,75,85,90]

method_dir = f'results_{args.algo}/'

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


    # 运行结果保存路径文件夹
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

    for iter in range(10):
        # args.seed = iter
        # np.random.seed(args.seed)
        # torch.manual_seed(args.seed)
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

            # 每代平均奖励曲线
        plt.figure()
        fig_output_filename1 = "/reward_{}_{}_{}.pdf".format(dataset, a, iter)
        plt.plot(local_reward_list, label="Reward")
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode mean reward curve')
        plt.savefig(pathname + fig_output_filename1)
        plt.close()

        plt.figure()
        fig_output_filename11 = "/reward_{}_{}_{}.eps".format(dataset, a, iter)
        plt.plot(local_reward_list, label="Reward")
        plt.xlabel('Episode')
        plt.ylabel(' Reward')
        plt.title('Episode mean reward curve')
        plt.savefig(pathname + fig_output_filename11)
        plt.close()


        # 每代平均损失
        plt.figure()
        fig_output_filename3 = "/policy_loss_{}_{}_{}.pdf".format(dataset, a, iter)
        plt.plot(local_loss_list, label="Policy loss")
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Episode policy loss curve')
        plt.savefig(pathname + fig_output_filename3)
        plt.close()

        plt.figure()
        fig_output_filename33 = "/policy_loss_{}_{}_{}.eps".format(dataset, a, iter)
        plt.plot(local_loss_list, label="Policy loss")
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Episode policy loss curve')
        plt.savefig(pathname + fig_output_filename33)
        plt.close()

        # 搜索完成
        output_filename_total = "/record_{}_{}_{}.txt".format(dataset, a,iter)
        f3 = open(pathname + output_filename_total, 'a+')
        f3.write('reward_list ={}\n'.format(str(local_reward_list)))

        f3.close()
        print("finished!")
