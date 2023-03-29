import os
from tqdm import tqdm
from models import utils as ut

dirname ='./data/sat_to_mdp/3sat/uf20-91/'
for fname in tqdm(os.listdir(dirname)):
    path = dirname+fname.replace(".txt","_ntable.txt")
    if os.path.exists(path): continue
    n,N,M,adj = ut.load_3sat(dirname+fname)
    n = adj.shape[0]
    ntable = ut.compute_ntable(adj)    
    f = open(path,"w+")
    for v in ntable.values():
        f.write(' '.join(list(map(str,v)))+'\n')
    f.close()