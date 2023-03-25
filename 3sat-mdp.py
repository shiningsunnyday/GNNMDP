import argparse 
import pickle
from pysat.solvers import Glucose3

parser = argparse.ArgumentParser()
parser.add_argument('--input_file',default="")
parser.add_argument('--output_file',default="")
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
            
    

with open(args.input_file, "r") as f:
    n, num_clauses, sat_clauses = read_sat(f)
    n, threesat_clauses = reduce_3sat(n, num_clauses, sat_clauses)
    g = Glucose3()
    for clause in threesat_clauses:
        g.add_clause(clause)
    
    print(g.solve() == int(args.input_file.split('sat=')[1][0]))

#     breakpoint()
#     mdp_prob = reduce_to_mdp(threesat_clauses)

# with open(args.output_file, "w+") as f:
#     write_mdp(mdp_prob)