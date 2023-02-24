
from models import utils as ut
from models import gnn
import torch.optim as optim
import dgl
import torch


def gnn_solver(args,mod,datapath,dataset1,dataset2,ts,train=True,model_path=None):
    adj, ntable = ut.load_data_adj_ntable(datapath, dataset1, dataset2)
    G = dgl.from_scipy(adj)
    if mod == 'gcn' or mod == 'gat':
        G = dgl.add_self_loop(G)
    # Model and optimizer
    in_feats, hid_feats, out_feats = args.noisy_size+1, args.hidden, args.hidden
    num_layers, input_dim, hidden_dim, output_dim = 3, out_feats, args.hidden, args.nclass

    if mod == 'sage':
        aggregator_type = 'gcn'
        model = gnn.SAGE(in_feats, hid_feats, out_feats, aggregator_type, num_layers,
                         input_dim, hidden_dim, output_dim)
    elif mod == 'gcn':
        model = gnn.GCN(in_feats, hid_feats, out_feats, num_layers,
                        input_dim, hidden_dim, output_dim)
    elif mod == 'gat':
        num_heads = 3
        model = gnn.GAT(in_feats, hid_feats, out_feats, num_heads,
                        num_layers, input_dim, hidden_dim, output_dim)
    elif mod == 'gin':
        aggregator_type = 'mean'
        fun_num_layers = 3
        model = gnn.GIN(in_feats, hid_feats, out_feats, aggregator_type,
                        fun_num_layers, num_layers, input_dim, hidden_dim, output_dim)
    elif mod == 'edge':
        model = gnn.EDGE(in_feats, hid_feats,out_feats,num_layers, input_dim, hidden_dim, output_dim)
    elif mod == 'tag':
        model = gnn.TAG(in_feats, hid_feats, out_feats, num_layers, input_dim, hidden_dim, output_dim)
    if train:
        step_size, gamma = 50, 0.5  # 50,0.5
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        model.train()
        local_ave_time, local_reward, local_loss, local_best_ind, \
        local_best_ind_panel_set, local_max_reward, local_best_state \
            = ts.train(args, model, ntable, G, optimizer=optimizer, scheduler=scheduler, train=True)

    else:

        model.load_state_dict(torch.load(model_path))
        local_ave_time, local_reward, local_loss, local_best_ind, \
        local_best_ind_panel_set, local_max_reward\
            = ts.train(args,model,ntable,G,optimizer=None,scheduler=None,train=False)

    if train:
        return local_ave_time, local_reward, local_loss, local_best_ind, \
               local_best_ind_panel_set, local_max_reward, local_best_state,ntable
    else:
        return local_ave_time, local_reward, local_loss, local_best_ind, \
               local_best_ind_panel_set, local_max_reward, ntable


def gnn_mdp(args,mod,datapath,dataset1,dataset2,ts,train=True,model_path=None):
    adj, ntable = ut.load_data_adj_ntable(datapath, dataset1, dataset2)
    G = dgl.from_scipy(adj)
    if mod == 'gcn' or mod == 'gat':
        G = dgl.add_self_loop(G)
    # Model and optimizer
    n = adj.shape[0]
    in_feats, hid_feats, out_feats = n+1, args.hidden, args.hidden
    num_layers, input_dim, hidden_dim, output_dim = 3, out_feats, args.hidden, n

    if mod == 'sage':
        aggregator_type = 'gcn'
        model = gnn.SAGE(in_feats, hid_feats, out_feats, aggregator_type, num_layers,
                         input_dim, hidden_dim, output_dim)
    elif mod == 'gcn':
        model = gnn.GCN(in_feats, hid_feats, out_feats, num_layers,
                        input_dim, hidden_dim, output_dim, mask=True)
    elif mod == 'gat':
        num_heads = 3
        model = gnn.GAT(in_feats, hid_feats, out_feats, num_heads,
                        num_layers, input_dim, hidden_dim, output_dim)
    elif mod == 'gin':
        aggregator_type = 'mean'
        fun_num_layers = 3
        model = gnn.GIN(in_feats, hid_feats, out_feats, aggregator_type,
                        fun_num_layers, num_layers, input_dim, hidden_dim, output_dim)
    elif mod == 'edge':
        model = gnn.EDGE(in_feats, hid_feats,out_feats,num_layers, input_dim, hidden_dim, output_dim)
    elif mod == 'tag':
        model = gnn.TAG(in_feats, hid_feats, out_feats, num_layers, input_dim, hidden_dim, output_dim)
    if train:
        step_size, gamma = 50, 0.5  # 50,0.5
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        model.train()
        local_ave_time, local_reward, local_loss, local_best_ind, \
        local_best_ind_panel_set, local_max_reward, local_best_state \
            = ts.train_gnn_mdp(args, model, ntable, G, optimizer=optimizer, scheduler=scheduler, train=True)

    else:

        model.load_state_dict(torch.load(model_path))
        local_ave_time, local_reward, local_loss, local_best_ind, \
        local_best_ind_panel_set, local_max_reward\
            = ts.train_gnn_mdp(args,model,ntable,G,optimizer=None,scheduler=None,train=False)

    if train:
        return local_ave_time, local_reward, local_loss, local_best_ind, \
               local_best_ind_panel_set, local_max_reward, local_best_state,ntable
    else:
        return local_ave_time, local_reward, local_loss, local_best_ind, \
               local_best_ind_panel_set, local_max_reward, ntable
