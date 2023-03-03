"""
Jian WU
wujian@sxufe.edu.cn
20/12/2020
"""
from __future__ import division
from __future__ import print_function
import time
import numpy as np
import torch
from models import utils as ut
from torch.nn import functional as F


def train(args,model,ntable,G,optimizer=None,scheduler=None,train=True):
    # Node features
    node_num= len(ntable)
    nr_degree = []
    for i in range(len(ntable)):
        nr_degree.append(len(ntable[i]))
    nr_degree = torch.tensor(nr_degree).reshape(-1,1)
    # nr_degree = torch.softmax(nr_degree,dim=0)
    # print(nr_degree)
    # degree = G.in_degrees().reshape(-1, 1)

    # 记录每代的最好个体，噪音，惩罚集
    # global_ind = []
    # global_panel_set = []
    # global_rewards = []
    best_state = []

    # 保存每代最优奖励
    episode_origin_mean_reward = []  # 保存每代平均奖励损失
    episode_loss = []
    global_best_ind = []
    global_best_ind_panel_set = []
    global_max_reward = -(node_num - 1)

    start = time.time()

    for epoch in range(1,args.epochs+1):
        print('Train and solve: ',epoch)
        o_features = nr_degree.expand(args.batch_size, -1, -1)
        # d_features = degree.expand(args.batch_size, -1, -1)
        # features = torch.zeros(args.batch_size, node_num, args.noisy_size)
        if args.algo == 'algo2':
            features = ut.sample1(args.batch_size, node_num, args.noisy_size)
        else:
            features = torch.eye(node_num).expand(args.batch_size, -1, -1)
        features = torch.cat([features,o_features],dim=-1)

        # 按批次 batch_size 计算每个顶点的出现的概率 p,是一个node_num行batch_size列的张量
        # """
        output = []
        if train:
            optimizer.zero_grad()
        for fea in features:
            out_temp = model(G,fea)
            output.append(out_temp)
        output = torch.cat(output,dim=1)

        # """
        # 通过一次伯努利实验对顶点进行采样 0 表示不在分辨集内，1表示在分辨集内

        set_vector, set_indicator = model.decide_action(output, args.batch_size)
        print(set_indicator.sum()/args.batch_size)


        reward_vector, \
        penal_vector, \
        local_best_ind, \
        local_best_panel, \
        local_best_reward = ut.reward(set_vector, ntable)

        print(local_best_reward)
        print("global:",global_max_reward)
        episode_origin_mean_reward.append(np.mean(reward_vector))
        # 记录最优个体与最优值
        if global_max_reward < local_best_reward and len(local_best_panel) == 0:  # 记录全局最好信息
            global_max_reward = local_best_reward
            global_best_ind = local_best_ind
            global_best_ind_panel_set = local_best_panel
            best_state = model.state_dict()
            # 保存整个网络
            # torch.save(model, PATH)
            # 保存网络中的参数, 速度快，占空间少
            # torch.save(model.state_dict(), PATH+model_save_file)
            # early_stop = 0
        # else:
        #     early_stop = early_stop +1
            # 针对上面一般的保存方法，加载的方法分别是：
            # model_dict = torch.load(PATH)
            # model_dict = model.load_state_dict(torch.load(PATH))
        # global_panel_set.append(global_best_ind_panel_set)
        # global_rewards.append(global_max_reward)
        # global_ind.append(global_best_ind)
        new_reward_vector = torch.FloatTensor(reward_vector)
        # reward_new = new_reward_vector

        l,lp= model.loss(set_indicator, new_reward_vector, output,epoch)
        if train:
            l.backward()
            optimizer.step()
            scheduler.step()

        episode_loss.append(lp.item())

    use_time = time.time()-start
    ave_time = use_time / args.epochs

    if train:
        return ave_time, episode_origin_mean_reward,episode_loss,global_best_ind,\
               global_best_ind_panel_set, global_max_reward,best_state
    else:
        return ave_time, episode_origin_mean_reward, episode_loss, global_best_ind, \
               global_best_ind_panel_set, global_max_reward

def train_gnn_mdp(args,model,ntable,G,optimizer=None,scheduler=None,train=True):
    # Node features
    node_num= len(ntable)
    nr_degree = []
    for i in range(len(ntable)):
        nr_degree.append(len(ntable[i]))
    nr_degree = torch.tensor(nr_degree).reshape(-1,1)
    # nr_degree = torch.softmax(nr_degree,dim=0)
    # print(nr_degree)
    # degree = G.in_degrees().reshape(-1, 1)

    # 记录每代的最好个体，噪音，惩罚集
    # global_ind = []
    # global_panel_set = []
    # global_rewards = []
    best_state = []

    # 保存每代最优奖励
    episode_origin_mean_reward = []  # 保存每代平均奖励损失
    episode_loss = []
    global_best_ind = []
    global_best_ind_panel_set = []
    global_max_reward = -(node_num - 1)

    start = time.time()

    for epoch in range(1,args.epochs+1):
        print('Train and solve: ',epoch)
        o_features = nr_degree.expand(args.batch_size, -1, -1)
        # d_features = degree.expand(args.batch_size, -1, -1)
        # features = torch.zeros(args.batch_size, node_num, args.noisy_size)
        features = (1-torch.eye(node_num)).expand(args.batch_size, -1, -1)
        features = torch.cat([features,o_features],dim=-1)

        # 按批次 batch_size 计算每个顶点的出现的概率 p,是一个node_num行batch_size列的张量
        # """
        output = []
        if train:
            optimizer.zero_grad()
        for fea in features:
            out_temp = model(G,fea)
            output.append(out_temp)
        output = torch.cat(output,dim=1) # masked from gnn model

        


        # determine membership set_vector
        set_vector, set_indicator = model.decide_action_mask(output, args.batch_size)

        print(set_indicator.sum())

        reward_vector, \
        penal_vector, \
        local_best_ind, \
        local_best_panel, \
        local_best_reward = ut.reward(set_vector, ntable)
        print(local_best_reward)
        print("global:",global_max_reward)

        episode_origin_mean_reward.append(np.mean(reward_vector))
        # 记录最优个体与最优值
        if global_max_reward < local_best_reward and len(local_best_panel) == 0:  # 记录全局最好信息
            global_max_reward = local_best_reward
            global_best_ind = local_best_ind
            global_best_ind_panel_set = local_best_panel
            best_state = model.state_dict()
            # 保存整个网络
            # torch.save(model, PATH)
            # 保存网络中的参数, 速度快，占空间少
            # torch.save(model.state_dict(), PATH+model_save_file)
            # early_stop = 0
        # else:
        #     early_stop = early_stop +1
            # 针对上面一般的保存方法，加载的方法分别是：
            # model_dict = torch.load(PATH)
            # model_dict = model.load_state_dict(torch.load(PATH))
        # global_panel_set.append(global_best_ind_panel_set)
        # global_rewards.append(global_max_reward)
        # global_ind.append(global_best_ind)
        new_reward_vector = torch.FloatTensor(reward_vector)
        # reward_new = new_reward_vector

        l = model.mask_loss(output, epoch)
        
        if train:
            l.backward()
            optimizer.step()
            scheduler.step()
        lp = model.loss(set_indicator, new_reward_vector, output.detach(),epoch)[1].detach()
        episode_loss.append(lp.item())

    use_time = time.time()-start
    ave_time = use_time / args.epochs

    if train:
        return ave_time, episode_origin_mean_reward,episode_loss,global_best_ind,\
               global_best_ind_panel_set, global_max_reward,best_state
    else:
        return ave_time, episode_origin_mean_reward, episode_loss, global_best_ind, \
               global_best_ind_panel_set, global_max_reward


def train_gnn_mvc(args,model,etable,G,optimizer=None,scheduler=None,train=True):
    # Node features
    node_num= len(etable)
    degree = []
    for i in range(len(etable)):
        degree.append(len(etable[i]))
    degree = torch.tensor(degree).reshape(-1,1)
  
    edge_mask, one_hot_node_features, edge_fea = ut.compute_edge_mask(G)
    best_state = []

    # 保存每代最优奖励
    episode_origin_mean_reward = []  # 保存每代平均奖励损失
    episode_loss = []
    global_best_ind = []
    global_best_ind_panel_set = []
    global_max_reward = -(node_num - 1)

    start = time.time()

    for epoch in range(1,args.epochs+1):
        print('Train and solve: ',epoch)
        o_features = degree.expand(args.batch_size, -1, -1)
        # d_features = degree.expand(args.batch_size, -1, -1)
        # features = torch.zeros(args.batch_size, node_num, args.noisy_size)
        features = one_hot_node_features.expand(args.batch_size, -1, -1)
        # features = torch.cat([features,o_features],dim=-1)

        # 按批次 batch_size 计算每个顶点的出现的概率 p,是一个node_num行batch_size列的张量
        # """
        output = []
        if train:
            optimizer.zero_grad()
        for fea in features:
            out_temp = model(G,fea,edge_fea)
            output.append(out_temp)
        output = torch.cat(output,dim=1) # masked from gnn model

        # determine membership set_vector
        set_vector, set_indicator = model.decide_action_mask(output, args.batch_size)

        print(set_indicator.sum())

        reward_vector, \
        penal_vector, \
        local_best_ind, \
        local_best_panel, \
        local_best_reward = ut.compute_mvc_reward(set_vector, etable)
        print(local_best_reward)
        print("global:",global_max_reward)

        episode_origin_mean_reward.append(np.mean(reward_vector))
        # 记录最优个体与最优值
        if global_max_reward < local_best_reward and len(local_best_panel) == 0:  # 记录全局最好信息
            global_max_reward = local_best_reward
            global_best_ind = local_best_ind
            global_best_ind_panel_set = local_best_panel
            best_state = model.state_dict()
            # 保存整个网络
            # torch.save(model, PATH)
            # 保存网络中的参数, 速度快，占空间少
            # torch.save(model.state_dict(), PATH+model_save_file)
            # early_stop = 0
        # else:
        #     early_stop = early_stop +1
            # 针对上面一般的保存方法，加载的方法分别是：
            # model_dict = torch.load(PATH)
            # model_dict = model.load_state_dict(torch.load(PATH))
        # global_panel_set.append(global_best_ind_panel_set)
        # global_rewards.append(global_max_reward)
        # global_ind.append(global_best_ind)
        new_reward_vector = torch.FloatTensor(reward_vector)
        # reward_new = new_reward_vector

        l = model.mask_loss(output, epoch, edge_mask=edge_mask)
        
        if train:
            l.backward()
            optimizer.step()
            scheduler.step()

        # not comparable
        lp = torch.tensor(0)
        episode_loss.append(lp.item())

    use_time = time.time()-start
    ave_time = use_time / args.epochs

    if train:
        return ave_time, episode_origin_mean_reward,episode_loss,global_best_ind,\
               global_best_ind_panel_set, global_max_reward,best_state
    else:
        return ave_time, episode_origin_mean_reward, episode_loss, global_best_ind, \
               global_best_ind_panel_set, global_max_reward
