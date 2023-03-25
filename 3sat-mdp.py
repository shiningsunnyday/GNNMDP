import argparse 
import pickle
from pysat.solvers import Glucose3
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--input_file',default="")
parser.add_argument('--output_dir',default="")
args = parser.parse_args()
    
def read_sat(f):
    clauses = []
    while True:
        line = f.readline()
        if not line: break
        if line.startswith("p cnf"):
            _, _, n, num_clauses = line.split()
        else:
            clauses.append(list(map(int, line.split()))[:-1])

    return int(n), int(num_clauses), clauses
    
def reduce_3sat(n, num_clauses, cnf):
    max_var = n+1
    new_cnf = []
    for clause in cnf:
        if len(clause) == 1:
            cur = clause[0]
            y1, y2 = max_var, max_var+1                     
            new_cnf.append([cur, y1, y2])
            new_cnf.append([cur, -y1, y2])
            new_cnf.append([cur, y1, -y2])
            new_cnf.append([cur, -y1, -y2])
            max_var += 2   
        elif len(clause) == 2:
            cur1, cur2 = clause
            y1 = max_var
            new_cnf.append([cur1,cur2,y1])
            new_cnf.append([cur1,cur2,-y1])
            max_var += 1
        elif len(clause) == 3:
            new_cnf.append(clause)
        else:
            k = len(clause)
            y = [max_var+i for i in range(k-3)]
            new_cnf.append([clause[0],clause[1],y[0]])
            for i in range(2, k-2):
                new_cnf.append([-y[i-2],clause[i],y[i-1]])
            new_cnf.append([-y[k-4],clause[k-2],clause[k-1]])
            max_var += k
    return max_var-1, new_cnf
    
    # print("p cnf %d %d" % (maxvar, len(new_cnf)))
    # for clause in new_cnf:
    #     print(" ".join([ "%d" % lit for lit in clause ]) + " 0")

    return new_cnf
            

def reduce_to_mdp(n, clauses):
    """
    for each var i: T_i, a_i, b_i1, F_i, b_i2, a_i2, i>=1
    for each clause j: c_j1, c_j2, c_j3, c_j4, c_j5, j>=0
    output:
    for each var i: [6*(i-1), 6*i-1]
    for each clause j: [6n+5j, 6n+5j+4]
    """
    m = len(clauses)
    adj = np.zeros((6*n+5*m,6*n+5*m))

    for i in range(1,n+1):
        for k in range(6):
            adj[6*(i-1)+k][6*(i-1)+(k+1)%6] = 1
            adj[6*(i-1)+(k+1)%6][6*(i-1)+k] = 1

    for j in range(m):
        for k in range(5):
            if k == 1: continue
            adj[6*n+5*j+1][6*n+5*j+k] = 1
            adj[6*n+5*j+k][6*n+5*j+1] = 1

    for j, clause in enumerate(clauses):
        for i in range(1, n+1):
            adj[6*(i-1)][6*n+5*j] = 1
            adj[6*n+5*j][6*(i-1)] = 1
    
            if i in clause:
                adj[6*(i-1)+3][6*n+5*j] = 1
                adj[6*n+5*j][6*(i-1)+3] = 1
                adj[6*(i-1)+3][6*n+5*j+2] = 1                
                adj[6*n+5*j+2][6*(i-1)+3] = 1

            if -i in clause:
                adj[6*(i-1)+3][6*n+5*j] = 1
                adj[6*n+5*j][6*(i-1)+3] = 1         
                adj[6*(i-1)][6*n+5*j+2] = 1                
                adj[6*n+5*j+2][6*(i-1)] = 1

            if (i not in clause) and (-i not in clause):
                adj[6*(i-1)+3][6*n+5*j+2] = 1                
                adj[6*n+5*j+2][6*(i-1)+3] = 1
                adj[6*(i-1)+3][6*n+5*j] = 1
                adj[6*n+5*j][6*(i-1)+3] = 1         
                adj[6*(i-1)][6*n+5*j+2] = 1                
                adj[6*n+5*j+2][6*(i-1)] = 1   

    return n, m, adj             



with open(args.input_file, "r") as f:
    n, num_clauses, sat_clauses = read_sat(f)  
    n, threesat_clauses = reduce_3sat(n, num_clauses, sat_clauses)
    g = Glucose3()
    for clause in threesat_clauses:
        g.add_clause(clause)
    
    # print(g.solve() == int(args.input_file.split('sat=')[1][0]))

    
    n, m, mdp_adj = reduce_to_mdp(n, threesat_clauses)

if args.output_dir:
    sat_file = args.input_file.split('/')[-1]
    path = args.output_dir + f'{sat_file}_adj_{6*n+5*m}_n={n}_m={m}.txt'
    with open(path, "w+") as f:
        for i in range(mdp_adj.shape[0]):
            for j in range(mdp_adj.shape[1]):
                if mdp_adj[i][j] == 1:
                    f.write(f"{i+1} {j+1} 1\n")
