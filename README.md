# LSMDP
# Datasets and visualization

(1) The adjacency of graph is generated by function of "generate_random_graph" in the package package of "generate_negtive_resolving_matrix_matlab.rar". And in 
this package, the negative resolving table is generated by the Matlab program of "*.m" such as "ntable_cluster_graph.m". Then all of the adjacency matrix and 
the corresponding negative tables are piut into the same directory "data". It is used as input for other algorithms.

(2) For visualization of the datasets,  some types of the datasets are printed in the screen.  Figure2 is generated by the function of "visual_graph.py".  

# An example for the basic implementation

## Example for solving MDP by algorithm2

In this case, we set some sentences in the file of "cmopute_dim.py" as follows:

(1)mod = 'gcn' ; (gcn, gin, sage, edge, tag are selected)

(2)train = True

(3) flag=2;  (1~6 numbers are selected)

Then the algorithm takes dataset in the root directory "data/random_gnm/adj_natable/", and output the calculated result into the directory "results".

## Example for solving MDP  by greedy repair policy only

In this case, we set some sentences in the file of "main_repair_only.py" or  the function of "compute_dim_repair_only.py"as follows:

(1)mod = 'gcn' ; (gcn, gin, sage, edge, tag are selected)

(2)train = False;

Then the algorithm takes dataset in the root directory "data/random_gnm/adj_natable/", and output the calculated result into the directory "results".

# Results generation
(1) The reseults of Dim1~Dim5 in Tables 1 and Tables 3 are obtained through extensive calculations by the python program of  "cmopute_dim.py".

(2)The results of Dim7 in Table1 and Table3 can be obtained by the Matlab Linear Programming Algorithm "LPsolv_main.m" in the package of 
  "LP_algo.zip", taking as input the adjacency matrix of graphs.
  
(3) The results of Dim6 in Table 1 and Table 3 can be obtained by the python function of "main_repaire_only.py" or the function of "compute_dim_repair_only.py". 

(4) According to Table1 and Table 3, the function of "comput_relative_ratio.py" gives the content of Table2 and Table4.

(5) Delete the repair module in the function of "compute_dim.py", one can get the results of columns 2,6 in Table5. 

The time columns 3,7 of Table5 can be got through "compute_ave_time.py". 
The columns of 4,8 can be obtained by remaining the repair policy module only in "compute_dim.py". 

(6)  Figure3 is generated through the module of "converge_analysis.py".

# Note

(1) We explore the posibality of learning to solve the metric dimension of graphs. The results may not be very strict. 
Using different GNN models, one may produce different results. Our first attempt is to learn the metric dimensions of a graph.
 The implementation of the code is one of our attempts. The output results may be different. Perhaps, one can improve it further.
 
(2) The code simply provides an implementation possibility. This work is partially motivated by the work of
 (R. Sato, M. Yamada, H. Kashima. Learning to Find Hard Instances of Graph Problems. arxiv, 2019. URL:https://arxiv.org/pdf/1902.09700v1.pdf). 
Particularly, according to the background of MDP, we design the specific reward function,data structure, and the repair policy to solve the MDP. 
Furthermore, we introduce the graph neural networks to solve the MDP.

# Program running system

The experiments are all implemented in the environment of Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz 2.21GHz 16.0GB. Pytorch, DGL and Matlab are needed. 
