import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



fig = plt.figure()
# sum
ax1= fig.add_subplot(231)
# 1
path1 = "data/random_cluster_graph/adj_natable/"
# dataset1: powerlaw_cluster_graph (brief:cluster)
d =5
dataset1 = "adj_100_{}_0.5.txt".format(d)
datapath =path1+dataset1
a = np.loadtxt(datapath)
row = np.reshape(a[:,0]-1,[len(a[:,0]),1])
col = np.reshape(a[:,1]-1,[len(a[:,0]),1])
data = np.reshape(a[:,2],[len(a[:,0]),1])
edgelist=[]
a = np.concatenate([row,col],axis=1)
for e1,e2 in a:
    edgelist.append((e1,e2))

G = nx.from_edgelist(a)
#pos = nx.spring_layout(G)
# pos = nx.spectral_layout(G)
# pos = nx.circular_layout(G)
pos = nx.kamada_kawai_layout(G)
nx.draw(G,pos,node_size=20,linewidths=0,alpha=0.7,width=1,edge_color='k')
plt.title('(a) PLCG',fontsize='xx-large',fontweight='heavy')

ax2= fig.add_subplot(232)
# 2
# dataset2 : random_powerlaw_tree (brief:powerlaw_tree)
path2 = "data/random_powerlaw_tree/adj_natable/"
gamma=1
dataset2 = "adj_100_{}.txt".format(gamma)
datapath =path2+dataset2
a = np.loadtxt(datapath)
row = np.reshape(a[:,0]-1,[len(a[:,0]),1])
col = np.reshape(a[:,1]-1,[len(a[:,0]),1])
data = np.reshape(a[:,2],[len(a[:,0]),1])
edgelist=[]
a = np.concatenate([row,col],axis=1)
for e1,e2 in a:
    edgelist.append((e1,e2))
G = nx.from_edgelist(a)
#pos = nx.spring_layout(G)
# pos = nx.spectral_layout(G)
# pos = nx.circular_layout(G)
pos = nx.kamada_kawai_layout(G)
nx.draw(G,pos,node_size=20,linewidths=0,alpha=0.7,width=1,edge_color='k')
plt.title('(b) RPLT',fontsize='xx-large',fontweight='heavy')

# 3
ax3= fig.add_subplot(233)
# dataset3 : connected_watts_strogatz_graph (brief:watts)
path3 = "data/random_watts/adj_natable/"
k=5
dataset3 = "adj_100_{}_0.5.txt".format(k)

datapath =path3+dataset3
a = np.loadtxt(datapath)
row = np.reshape(a[:,0]-1,[len(a[:,0]),1])
col = np.reshape(a[:,1]-1,[len(a[:,0]),1])
data = np.reshape(a[:,2],[len(a[:,0]),1])
edgelist=[]
a = np.concatenate([row,col],axis=1)
for e1,e2 in a:
    edgelist.append((e1,e2))
G = nx.from_edgelist(a)
#pos = nx.spring_layout(G)
# pos = nx.spectral_layout(G)
# pos = nx.circular_layout(G)
pos = nx.kamada_kawai_layout(G)
nx.draw(G,pos,node_size=20,linewidths=0,alpha=0.7,width=1,edge_color='k')
plt.title('(c) WATTS',fontsize='xx-large',fontweight='heavy')


# 4
ax4= fig.add_subplot(234)
# dataset4 : random_gnm (brief:gnm)
path4 = "data/random_gnm/adj_natable/"
m=250
dataset4 = "adj_100_{}.txt".format(m)
datapath = path4+dataset4
a = np.loadtxt(datapath)
row = np.reshape(a[:,0]-1,[len(a[:,0]),1])
col = np.reshape(a[:,1]-1,[len(a[:,0]),1])
data = np.reshape(a[:,2],[len(a[:,0]),1])
edgelist=[]
a = np.concatenate([row,col],axis=1)
for e1,e2 in a:
    edgelist.append((e1,e2))
G = nx.from_edgelist(a)
#pos = nx.spring_layout(G)
# pos = nx.spectral_layout(G)
# pos = nx.circular_layout(G)
pos = nx.kamada_kawai_layout(G)
nx.draw(G,pos,node_size=20,linewidths=0,alpha=0.7,width=0.1,edge_color='k')
plt.title('(d) GNM',fontsize='xx-large',fontweight='heavy')


# 5
ax5= fig.add_subplot(235)
# dataset5 : random_gnp (brief:gnp)
path5 = "data/random_gnp/adj_natable/"
p=0.05
dataset5 = "adj_100_{}.txt".format(p)
datapath = path5+dataset5
a = np.loadtxt(datapath)
row = np.reshape(a[:,0]-1,[len(a[:,0]),1])
col = np.reshape(a[:,1]-1,[len(a[:,0]),1])
data = np.reshape(a[:,2],[len(a[:,0]),1])
edgelist=[]
a = np.concatenate([row,col],axis=1)
for e1,e2 in a:
    edgelist.append((e1,e2))
G = nx.from_edgelist(a)
#pos = nx.spring_layout(G)
# pos = nx.spectral_layout(G)
# pos = nx.circular_layout(G)
pos = nx.kamada_kawai_layout(G)
nx.draw(G,pos,node_size=20,linewidths=0,alpha=0.7,width=1,edge_color='k')
plt.title('(e) GNP',fontsize='xx-large',fontweight='heavy')

# 6
ax6= fig.add_subplot(236)
# dataset6 : random_regular_graph (brief:regular)
path6 = "data/random_regular_graph/adj_natable/"
d=5
dataset6 = "adj_100_{}.txt".format(d)
datapath = path6+dataset6
a = np.loadtxt(datapath)
row = np.reshape(a[:,0]-1,[len(a[:,0]),1])
col = np.reshape(a[:,1]-1,[len(a[:,0]),1])
data = np.reshape(a[:,2],[len(a[:,0]),1])
edgelist=[]
a = np.concatenate([row,col],axis=1)
for e1,e2 in a:
    edgelist.append((e1,e2))
G = nx.from_edgelist(a)
#pos = nx.spring_layout(G)
# pos = nx.spectral_layout(G)
# pos = nx.circular_layout(G)
pos = nx.kamada_kawai_layout(G)
nx.draw(G,pos,node_size=20,linewidths=0,alpha=0.7,width=1,edge_color='k')
plt.title('(f) RRG',fontsize='xx-large',fontweight='heavy')
plt.show()




