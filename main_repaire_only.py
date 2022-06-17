import os
import argparse
import gnn_solver as gs
from models import train_gcn as ts
import torch
import numpy as np
import repair_method as rm
from models import utils as ut


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=13, help='Random seed.')  # 456
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--Early_stop', type=int, default=101,
                    help='Early_stop.')
#0.5
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-3,
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


mod = 'gcn'  # gcn,gat,sage,gin,edge,tag
train = False
# data
dataset = 'powerlawtree'
d=[10] #[10,9,8,7,6,5,4,3,2,1]  # [1,2,3,4,5,6,7,8,9,10]
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


    model_save_file = '\parameter_{}.pkl'.format(mod)
    model_save_path_file = model_save_path + model_save_file
    adj, ntable = ut.load_data_adj_ntable(datapath, dataset1, dataset2)


    print("Begin to repair!")
    iter =args.epochs

    global_resolving_set = list(range(len(ntable)))
    r_set,dim = rm.repair_iter(args,global_resolving_set, ntable, iter)
    record_repair_path = "/{}_{}_record_repair.txt".format(dataset, mod)
    f = open(pathname + record_repair_path, 'a+')
    f.write('global_dim ={}\n'.format(dim))
    f.write('resolving_set ={}\n'.format(str(r_set)))
    f.close()














