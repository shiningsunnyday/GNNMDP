
from models import utils as ut
from models import gnn
import numpy as np
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
    if mod == 'distmask':
        assert (args.mask_c>0) ^ args.do_omp
        if args.do_omp:
            # rew_func = lambda x: ut.reward(x,ntable)[-1] # return best_reward
            rew_func = lambda x, s: -args.distmask_c*x.sum()/x.shape[-1]+s
        else:
            rew_func = None
        model = gnn.DistMask(output_dim, mask_c=args.mask_c, do_omp=rew_func)
    elif mod == 'sage':
        aggregator_type = 'gcn'
        model = gnn.SAGE(in_feats, hid_feats, out_feats, aggregator_type, num_layers,
                         input_dim, hidden_dim, output_dim, num_hidden_layers=args.num_hidden_layers, mask_c=args.mask_c)
    elif mod == 'gcn':
        model = gnn.GCN(in_feats, hid_feats, out_feats, num_layers,
                        input_dim, hidden_dim, output_dim, num_hidden_layers=args.num_hidden_layers, mask_c=args.mask_c)
    elif mod == 'gat':
        num_heads = 3
        model = gnn.GAT(in_feats, hid_feats, out_feats, num_heads,
                        num_layers, input_dim, hidden_dim, output_dim)
    elif mod == 'gin':
        aggregator_type = 'mean'
        fun_num_layers = 3
        model = gnn.GIN(in_feats, hid_feats, out_feats, aggregator_type,
                        fun_num_layers, num_layers, input_dim, hidden_dim, output_dim, num_hidden_layers=args.num_hidden_layers, mask_c=args.mask_c)
    elif mod == 'edge':
        model = gnn.EDGE(in_feats, hid_feats,out_feats,num_layers, input_dim, hidden_dim, output_dim, num_hidden_layers=args.num_hidden_layers, mask_c=args.mask_c)
    elif mod == 'tag':
        model = gnn.TAG(in_feats, hid_feats, out_feats, num_layers, input_dim, hidden_dim, output_dim, num_hidden_layers=args.num_hidden_layers, mask_c=args.mask_c)
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


def gnn_mvc(args,mod,datapath,dataset1,dataset2,ts,train=True,model_path=None):
    adj, _ = ut.load_data_adj_ntable(datapath, dataset1, dataset2)    
    etable, m = ut.compute_etable(adj)

    G = dgl.from_scipy(adj)
    if mod == 'gcn' or mod == 'gat':
        G = dgl.add_self_loop(G)
    n = G.num_nodes()
    # Model and optimizer
    in_feats, hid_feats, out_feats = m, args.hidden, args.hidden
    num_layers, input_dim, hidden_dim, output_dim = 3, out_feats, args.hidden, n

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
    elif mod == 'gine':
        hid_feats = m
        out_feats = m
        input_dim = m
        fun_num_layers = 3
        # keep hidden_dim the same, used for last mlp
        model = gnn.GINE(in_feats, hid_feats, out_feats,
                 fun_num_layers,num_layers, input_dim, hidden_dim, output_dim, mask_c=args.mask_c)
    elif mod == 'tag':
        model = gnn.TAG(in_feats, hid_feats, out_feats, num_layers, input_dim, hidden_dim, output_dim)
    if train:
        step_size, gamma = 50, 0.5  # 50,0.5
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        model.train()
        local_ave_time, local_reward, local_loss, local_best_ind, \
        local_best_ind_panel_set, local_max_reward, local_best_state \
            = ts.train_gnn_mvc(args, model, etable, G, optimizer=optimizer, scheduler=scheduler, train=True)

    else:

        model.load_state_dict(torch.load(model_path))
        local_ave_time, local_reward, local_loss, local_best_ind, \
        local_best_ind_panel_set, local_max_reward\
            = ts.train_gnn_mvc(args,model,etable,G,optimizer=None,scheduler=None,train=False)

    if train:
        return local_ave_time, local_reward, local_loss, local_best_ind, \
               local_best_ind_panel_set, local_max_reward, local_best_state,etable
    else:
        return local_ave_time, local_reward, local_loss, local_best_ind, \
               local_best_ind_panel_set, local_max_reward, etable


def gnn_bisect(args,mod,datapath,dataset1,dataset2,ts,train=True,model_path=None):
    adj, _ = ut.load_data_adj_ntable(datapath, dataset1, dataset2)    
    etable, m = ut.compute_etable(adj)

    G = dgl.from_scipy(adj)
    if mod == 'gcn' or mod == 'gat':
        G = dgl.add_self_loop(G)
    n = G.num_nodes()
    # Model and optimizer
    in_feats, hid_feats, out_feats = m, args.hidden, args.hidden
    num_layers, input_dim, hidden_dim, output_dim = 3, out_feats, args.hidden, n

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
    elif mod == 'gine':
        hid_feats = m
        out_feats = m
        input_dim = m
        fun_num_layers = 3
        # keep hidden_dim the same, used for last mlp
        model = gnn.GINE(in_feats, hid_feats, out_feats,
                 fun_num_layers,num_layers, input_dim, hidden_dim, output_dim, mask_c=args.mask_c)
    elif mod == 'tag':
        model = gnn.TAG(in_feats, hid_feats, out_feats, num_layers, input_dim, hidden_dim, output_dim)
    if train:
        step_size, gamma = 50, 0.5  # 50,0.5
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        model.train()
        local_ave_time, local_reward, local_loss, local_best_ind, \
        local_best_ind_panel_set, local_max_reward, local_best_state \
            = ts.train_gnn_bisect(args, model, etable, G, optimizer=optimizer, scheduler=scheduler, train=True)
    else:
        model.load_state_dict(torch.load(model_path))
        local_ave_time, local_reward, local_loss, local_best_ind, \
        local_best_ind_panel_set, local_max_reward\
            = ts.train_gnn_bisect(args,model,etable,G,optimizer=None,scheduler=None,train=False)

    if train:
        return local_ave_time, local_reward, local_loss, local_best_ind, \
               local_best_ind_panel_set, local_max_reward, local_best_state,etable
    else:
        return local_ave_time, local_reward, local_loss, local_best_ind, \
               local_best_ind_panel_set, local_max_reward, etable


def gnn_dom_k(args,mod,datapath,dataset1,dataset2,ts,train=True,model_path=None):
    adj, _ = ut.load_data_adj_ntable(datapath, dataset1, dataset2)    
    ktable = ut.compute_ktable(adj,args.dom_k) # {node:set(nodes that are distance>k away)}

    G = dgl.from_scipy(adj)
    if mod == 'gcn' or mod == 'gat':
        G = dgl.add_self_loop(G)
    n = G.num_nodes()
    # Model and optimizer
    in_feats, hid_feats, out_feats = n+1, args.hidden, args.hidden
    num_layers, input_dim, hidden_dim, output_dim = 3, out_feats, args.hidden, n

    if mod == 'sage':
        aggregator_type = 'gcn'
        model = gnn.SAGE(in_feats, hid_feats, out_feats, aggregator_type, num_layers,
                         input_dim, hidden_dim, output_dim, num_hidden_layers=args.num_hidden_layers, mask_c=args.mask_c)
    elif mod == 'gcn':
        model = gnn.GCN(in_feats, hid_feats, out_feats, num_layers,
                        input_dim, hidden_dim, output_dim, num_hidden_layers=args.num_hidden_layers, mask_c=args.mask_c)
    elif mod == 'gat':
        num_heads = 3
        model = gnn.GAT(in_feats, hid_feats, out_feats, num_heads,
                        num_layers, input_dim, hidden_dim, output_dim)
    elif mod == 'gin':
        aggregator_type = 'mean'
        fun_num_layers = 3
        model = gnn.GIN(in_feats, hid_feats, out_feats, aggregator_type,
                        fun_num_layers, num_layers, input_dim, hidden_dim, output_dim, num_hidden_layers=args.num_hidden_layers, mask_c=args.mask_c)
    elif mod == 'edge':
        model = gnn.EDGE(in_feats, hid_feats,out_feats,num_layers, input_dim, hidden_dim, output_dim, num_hidden_layers=args.num_hidden_layers, mask_c=args.mask_c)
    elif mod == 'tag':
        model = gnn.TAG(in_feats, hid_feats, out_feats, num_layers, input_dim, hidden_dim, output_dim, num_hidden_layers=args.num_hidden_layers, mask_c=args.mask_c)

    if train:
        step_size, gamma = 50, 0.5  # 50,0.5
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        model.train()
        local_ave_time, local_reward, local_loss, local_best_ind, \
        local_best_ind_panel_set, local_max_reward, local_best_state \
            = ts.train_gnn_dom_k(args, model, ktable, G, optimizer=optimizer, scheduler=scheduler, train=True)
    else:
        model.load_state_dict(torch.load(model_path))
        local_ave_time, local_reward, local_loss, local_best_ind, \
        local_best_ind_panel_set, local_max_reward\
            = ts.train_gnn_dom_k(args,model,ktable,G,optimizer=None,scheduler=None,train=False)

    if train:
        return local_ave_time, local_reward, local_loss, local_best_ind, \
               local_best_ind_panel_set, local_max_reward, local_best_state,ktable
    else:
        return local_ave_time, local_reward, local_loss, local_best_ind, \
               local_best_ind_panel_set, local_max_reward, ktable


def gnn_steiner(args,mod,datapath,dataset1,dataset2,ts,train=True,model_path=None):
    adj, T, opt = ut.load_steinerlib(dataset1, datapath)
    ttable = ut.compute_ttable(adj,T)
    G = dgl.from_scipy(adj)
    if mod == 'gcn' or mod == 'gat':
        G = dgl.add_self_loop(G)
    n = G.num_nodes()
    # Model and optimizer
    in_feats, hid_feats, out_feats = n, args.hidden, args.hidden
    num_layers, input_dim, hidden_dim, output_dim = 3, out_feats, args.hidden, n

    if mod == 'sage':
        aggregator_type = 'gcn'
        model = gnn.SAGE(in_feats, hid_feats, out_feats, aggregator_type, num_layers,
                         input_dim, hidden_dim, output_dim, num_hidden_layers=args.num_hidden_layers, mask_c=args.mask_c)
    elif mod == 'gcn':
        model = gnn.GCN(in_feats, hid_feats, out_feats, num_layers,
                        input_dim, hidden_dim, output_dim, num_hidden_layers=args.num_hidden_layers, mask_c=args.mask_c, act='relu')
    elif mod == 'gat':
        num_heads = 3
        model = gnn.GAT(in_feats, hid_feats, out_feats, num_heads,
                        num_layers, input_dim, hidden_dim, output_dim)
    elif mod == 'gin':
        aggregator_type = 'mean'
        fun_num_layers = 3
        model = gnn.GIN(in_feats, hid_feats, out_feats, aggregator_type,
                        fun_num_layers, num_layers, input_dim, hidden_dim, output_dim, num_hidden_layers=args.num_hidden_layers, mask_c=args.mask_c)
    elif mod == 'edge':
        model = gnn.EDGE(in_feats, hid_feats,out_feats,num_layers, input_dim, hidden_dim, output_dim, num_hidden_layers=args.num_hidden_layers, mask_c=args.mask_c)
    elif mod == 'tag':
        model = gnn.TAG(in_feats, hid_feats, out_feats, num_layers, input_dim, hidden_dim, output_dim, num_hidden_layers=args.num_hidden_layers, mask_c=args.mask_c)

    if train:
        step_size, gamma = 50, 0.5  # 50,0.5
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        model.train()
        local_ave_time, local_reward, local_loss, local_best_ind, \
        local_best_ind_panel_set, local_max_reward, local_best_state \
            = ts.train_steiner(args, model, ttable, adj.tocsr(), G, optimizer=optimizer, scheduler=scheduler, train=True)
    else:
        model.load_state_dict(torch.load(model_path))
        local_ave_time, local_reward, local_loss, local_best_ind, \
        local_best_ind_panel_set, local_max_reward\
            = ts.train_steiner(args,model,ttable,adj.tocsr(),G,optimizer=None,scheduler=None,train=False)

    if train:
        return local_ave_time, local_reward, local_loss, local_best_ind, \
               local_best_ind_panel_set, local_max_reward, local_best_state,None
    else:
        return local_ave_time, local_reward, local_loss, local_best_ind, \
               local_best_ind_panel_set, local_max_reward, None


