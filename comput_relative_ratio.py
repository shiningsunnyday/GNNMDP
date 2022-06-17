import  torch

# gnm_rho = []
# gnm_gcn_dim = torch.tensor([7,7,8,9,8,9,9,11,12,12]).reshape(-1,1)
# gnm_gin_dim = torch.tensor([7,8,7,9,9,10,10,11,11,14]).reshape(-1,1)
# gnm_sage_dim = torch.tensor([7,9,8,9,10,9,10,13,12,14]).reshape(-1,1)
# gnm_edge_dim = torch.tensor([7,7,8,9,10,10,10,11,12,13]).reshape(-1,1)
# gnm_tag_dim = torch.tensor([7,8,8,8,9,9,10,11,12,13]).reshape(-1,1)
# gnm_greed_dim = torch.tensor([8,8,9,9,9,10,11,13,13,14]).reshape(-1,1)
# gnm_lp_dim = torch.tensor([8,8,8,9,9,10,11,12,12,14]).reshape(-1,1)
#
# gnm_rho_gcn = torch.mean(gnm_gcn_dim/gnm_lp_dim,dim=0).numpy()[0]
# gnm_rho.append(gnm_rho_gcn)
# gnm_rho_gin = torch.mean(gnm_gin_dim/gnm_lp_dim,dim=0).numpy()[0]
# gnm_rho.append(gnm_rho_gin)
# gnm_rho_sage = torch.mean(gnm_sage_dim/gnm_lp_dim,dim=0).numpy()[0]
# gnm_rho.append(gnm_rho_sage)
# gnm_rho_edge = torch.mean(gnm_edge_dim/gnm_lp_dim,dim=0).numpy()[0]
# gnm_rho.append(gnm_rho_edge)
# gnm_rho_tag = torch.mean(gnm_tag_dim/gnm_lp_dim,dim=0).numpy()[0]
# gnm_rho.append(gnm_rho_tag)
# gnm_rho_greed = torch.mean(gnm_greed_dim/gnm_lp_dim,dim=0).numpy()[0]
# gnm_rho.append(gnm_rho_greed)
# print('gnm_rho=',gnm_rho)

# gnp_rho = []
# gnp_gcn_dim = torch.tensor([7,9,14,12,10,10,10,12,15,22]).reshape(-1,1)
# gnp_gin_dim = torch.tensor([6,9,15,12,10,10,10,12,15,23]).reshape(-1,1)
# gnp_sage_dim = torch.tensor([7,9,14,12,11,10,10,11,14,23]).reshape(-1,1)
# gnp_edge_dim = torch.tensor([7,9,14,13,10,10,11,14,16,23]).reshape(-1,1)
# gnp_tag_dim = torch.tensor([7,9,14,12,10,10,11,12,16,23]).reshape(-1,1)
# gnp_greed_dim = torch.tensor([8,11,14,13,10,10,11,12,15,23]).reshape(-1,1)
# gnp_lp_dim = torch.tensor([5,9,16,12,11,10,11,13,13,18]).reshape(-1,1)

# gnp_rho_gcn = torch.mean(gnp_gcn_dim/gnp_lp_dim,dim=0).numpy()[0]
# gnp_rho.append(gnp_rho_gcn)
# gnp_rho_gin = torch.mean(gnp_gin_dim/gnp_lp_dim,dim=0).numpy()[0]
# gnp_rho.append(gnp_rho_gin)
# gnp_rho_sage = torch.mean(gnp_sage_dim/gnp_lp_dim,dim=0).numpy()[0]
# gnp_rho.append(gnp_rho_sage)
# gnm_rho_edge = torch.mean(gnp_edge_dim/gnp_lp_dim,dim=0).numpy()[0]
# gnp_rho.append(gnm_rho_edge)
# gnp_rho_tag = torch.mean(gnp_tag_dim/gnp_lp_dim,dim=0).numpy()[0]
# gnp_rho.append(gnp_rho_tag)
# gnp_rho_greed = torch.mean(gnp_greed_dim/gnp_lp_dim,dim=0).numpy()[0]
# gnp_rho.append(gnp_rho_greed)
# print('gnp_rho=',gnp_rho)


# rrg_rho = []
# rrg_gcn_dim = torch.tensor([6,8,8,8,14,14,13,12,11,10]).reshape(-1,1)
# rrg_gin_dim = torch.tensor([7,9,9,9,13,14,13,12,11,10]).reshape(-1,1)
# rrg_sage_dim = torch.tensor([8,9,8,9,14,15,12,12,10,11]).reshape(-1,1)
# rrg_edge_dim = torch.tensor([8,9,8,9,13,15,14,12,11,10]).reshape(-1,1)
# rrg_tag_dim = torch.tensor([8,8,9,9,13,15,13,12,11,11]).reshape(-1,1)
# rrg_greed_dim = torch.tensor([7,9,9,9,14,15,14,13,11,11]).reshape(-1,1)
# rrg_lp_dim = torch.tensor([7,8,8,9,15,16,14,12,11,10]).reshape(-1,1)
#
# rrg_rho_gcn = torch.mean(rrg_gcn_dim/rrg_lp_dim,dim=0).numpy()[0]
# rrg_rho.append(rrg_rho_gcn)
# rrg_rho_gin = torch.mean(rrg_gin_dim/rrg_lp_dim,dim=0).numpy()[0]
# rrg_rho.append(rrg_rho_gin)
# rrg_rho_sage = torch.mean(rrg_sage_dim/rrg_lp_dim,dim=0).numpy()[0]
# rrg_rho.append(rrg_rho_sage)
# rrg_rho_edge = torch.mean(rrg_edge_dim/rrg_lp_dim,dim=0).numpy()[0]
# rrg_rho.append(rrg_rho_edge)
# rrg_rho_tag = torch.mean(rrg_tag_dim/rrg_lp_dim,dim=0).numpy()[0]
# rrg_rho.append(rrg_rho_tag)
# rrg_rho_greed = torch.mean(rrg_greed_dim/rrg_lp_dim,dim=0).numpy()[0]
# rrg_rho.append(rrg_rho_greed)
# print('rrg_rho=',rrg_rho)


# plcg_rho = []
# plcg_gcn_dim = torch.tensor([10,13,12,13,12,13,13,22,27,47]).reshape(-1,1)
# plcg_gin_dim = torch.tensor([12,14,14,13,13,10,13,20,28,43]).reshape(-1,1)
# plcg_sage_dim = torch.tensor([10,14,13,13,12,12,14,20,26,40]).reshape(-1,1)
# plcg_edge_dim = torch.tensor([10,13,14,14,13,12,13,21,25,43]).reshape(-1,1)
# plcg_tag_dim = torch.tensor([11,17,15,12,11,12,13,19,30,42]).reshape(-1,1)
# plcg_greed_dim = torch.tensor([12,17,15,14,12,13,14,20,32,46]).reshape(-1,1)
# plcg_lp_dim = torch.tensor([8,17,14,14,12,9,9,14,21,36]).reshape(-1,1)
#
# plcg_rho_gcn = torch.mean(plcg_gcn_dim/plcg_lp_dim,dim=0).numpy()[0]
# plcg_rho.append(plcg_rho_gcn)
# plcg_rho_gin = torch.mean(plcg_gin_dim/plcg_lp_dim,dim=0).numpy()[0]
# plcg_rho.append(plcg_rho_gin)
# plcg_rho_sage = torch.mean(plcg_sage_dim/plcg_lp_dim,dim=0).numpy()[0]
# plcg_rho.append(plcg_rho_sage)
# plcg_rho_edge = torch.mean(plcg_edge_dim/plcg_lp_dim,dim=0).numpy()[0]
# plcg_rho.append(plcg_rho_edge)
# plcg_rho_tag = torch.mean(plcg_tag_dim/plcg_lp_dim,dim=0).numpy()[0]
# plcg_rho.append(plcg_rho_tag)
# rrg_rho_greed = torch.mean(plcg_greed_dim/plcg_lp_dim,dim=0).numpy()[0]
# plcg_rho.append(rrg_rho_greed)
# print('plcg_rho=',plcg_rho)

# plt_rho = []
# plt_gcn_dim = torch.tensor([31,31,38,31,39,40,41,50,33,44]).reshape(-1,1)
# plt_gin_dim = torch.tensor([31,31,35,31,38,40,41,56,34,42]).reshape(-1,1)
# plt_sage_dim = torch.tensor([31,31,36,31,37,39,39,52,33,41]).reshape(-1,1)
# plt_edge_dim = torch.tensor([31,31,35,31,37,40,40,52,32,42]).reshape(-1,1)
# plt_tag_dim = torch.tensor([31,31,35,31,37,42,40,54,33,41]).reshape(-1,1)
# plt_greed_dim = torch.tensor([31,31,35,31,37,39,39,50,32,41]).reshape(-1,1)
# plt_lp_dim = torch.tensor([31,31,35,31,37,39,39,50,32,41]).reshape(-1,1)
#
# plt_rho_gcn = torch.mean(plt_gcn_dim/plt_lp_dim,dim=0).numpy()[0]
# plt_rho.append(plt_rho_gcn)
# plcg_rho_gin = torch.mean(plt_gin_dim/plt_lp_dim,dim=0).numpy()[0]
# plt_rho.append(plcg_rho_gin)
# plt_rho_sage = torch.mean(plt_sage_dim/plt_lp_dim,dim=0).numpy()[0]
# plt_rho.append(plt_rho_sage)
# plcg_rho_edge = torch.mean(plt_edge_dim/plt_lp_dim,dim=0).numpy()[0]
# plt_rho.append(plcg_rho_edge)
# plt_rho_tag = torch.mean(plt_tag_dim/plt_lp_dim,dim=0).numpy()[0]
# plt_rho.append(plt_rho_tag)
# plt_rho_greed = torch.mean(plt_greed_dim/plt_lp_dim,dim=0).numpy()[0]
# plt_rho.append(plt_rho_greed)
# print('plt_rho=',plt_rho)

watts_rho = []
watts_gcn_dim = torch.tensor([6,9,14,12,10,10,10,13,17,25]).reshape(-1,1)
watts_gin_dim = torch.tensor([6,9,14,12,10,10,11,13,20,24]).reshape(-1,1)
watts_sage_dim = torch.tensor([7,9,14,12,10,10,10,14,19,25]).reshape(-1,1)
watts_edge_dim = torch.tensor([7,9,14,12,11,10,11,13,19,27]).reshape(-1,1)
watts_tag_dim = torch.tensor([8,9,14,13,11,11,11,13,20,25]).reshape(-1,1)
watts_greed_dim = torch.tensor([6,10,14,12,11,10,10,15,17,26]).reshape(-1,1)
watts_lp_dim = torch.tensor([5,5,15,13,11,10,11,14,20,21]).reshape(-1,1)

watts_rho_gcn = torch.mean(watts_gcn_dim/watts_lp_dim,dim=0).numpy()[0]
watts_rho.append(watts_rho_gcn)
watts_rho_gin = torch.mean(watts_gin_dim/watts_lp_dim,dim=0).numpy()[0]
watts_rho.append(watts_rho_gin)
watts_rho_sage = torch.mean(watts_sage_dim/watts_lp_dim,dim=0).numpy()[0]
watts_rho.append(watts_rho_sage)
watts_rho_edge = torch.mean(watts_edge_dim/watts_lp_dim,dim=0).numpy()[0]
watts_rho.append(watts_rho_edge)
watts_rho_tag = torch.mean(watts_tag_dim/watts_lp_dim,dim=0).numpy()[0]
watts_rho.append(watts_rho_tag)
watts_rho_greed = torch.mean(watts_greed_dim/watts_lp_dim,dim=0).numpy()[0]
watts_rho.append(watts_rho_greed)
print('watts_rho=',watts_rho)

