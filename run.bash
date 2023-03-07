#!/bin/bash

algos='gnn-mdp'
flags='6 5 4 3 2'
lambdas='0.03 0.1 0.3 1.0 3.0 10.0 30.0'
lrs='0.01'
for algo in $algos
do
    for flag in $flags
    do
        for lambda in $lambdas
        do
            echo $algo;
            echo $flag;
            echo $lambda;
            python compute_dim.py --batch_size 1 --mask_c $lambda --algo gnn-mdp --flag $flag --lr 0.01 --epochs 1000 --num_iters 1;
            # python compute_dim.py --flag $flag --algo algo2 --lr 0.01;        
        done  
    done
done

