import os
import argparse
import gnn_solver as gs
import torch
import numpy as np
from models import utils as ut

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=456, help='Random seed.')  # 456,13
parser.add_argument('--epochs', type=int, default=100,
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
parser.add_argument('--nclass', type=int, default=1,
                    help='Number of output units.')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Number of batch size for training.')
parser.add_argument('--noisy_size', type=int, default=16,
                    help='Number of batch size for training.')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()

# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# os.environ["PYTHONHASHSEED"] = str(args.seed)

mod = 'gcn'  # gcn,gat,sage,gin,edge,tag
train = False
if not train:
    import repair_method as rm

if mod == 'gcn':
    from models import train_gcn as ts   # train_gcn,train_sage, train_gin,train_gat
elif mod == 'gat':
    from models import train_gat as ts
elif mod == 'sage':
    from models import train_sage as ts
elif mod == 'gin':
    from models import train_gin as ts
elif mod == 'edge':
    from models import train_edge as ts
elif mod == 'tag':
    from models import train_tag as ts
# data
dataset = 'powerlawtree'
d= [10]#[10,9,8,7,6,5,4,3,2,1]  # [1,2,3,4,5,6,7,8,9,10]
for i in d:
    a = i
    datapath = "data/random_powerlaw_tree/adj_natable/"
    dataset1 = "adj_100_{}.txt".format(a)
    dataset2 = "ntable_100_{}.txt".format(a)


    # 运行结果保存路径文件夹
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

    for iter in range(10):
        # args.seed = iter
        # np.random.seed(args.seed)
        # torch.manual_seed(args.seed)
        print('Iteration: ', iter+1)

        # features = ut.sample1(args.batch_size, 100, args.noisy_size)
        # print(features)
        if train:
            local_ave_time, local_reward_list, local_loss_list, local_best_ind, \
            local_best_ind_panel_set, local_max_reward, local_best_state \
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
    if train:
        torch.save(global_state, model_save_path_file)
    # 每代运行结果
    if train:
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
        # 搜索完成

        print("finished!")
    else:
        record_path_test = "/{}_{}_record_test.txt".format(dataset, mod)
        f = open(pathname + record_path_test, 'a+')
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
        # 搜索完成
        print("finished!")

    if not train:
        print("Begin to repair!")
        iter = 200  # len(global_resolving_set)
        r_set, dim = rm.repair_iter(args, global_resolving_set, ntable, iter)
        record_repair_path = "/{}_{}_record_repair.txt".format(dataset, mod)
        f = open(pathname + record_repair_path, 'a+')
        f.write('global_dim ={}\n'.format(dim))
        f.write('resolving_set ={}\n'.format(str(r_set)))
        f.close()











