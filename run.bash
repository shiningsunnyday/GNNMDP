#!/bin/bash

algos='gnn-mdp'
flags='5 1 2 3 4 6'
lambda='2.0 0.03 0.1 0.3 1.0 3.0 10.0'
for flag in $flags
do
    for algo in $algos
    do
        if [ $algo = 'gnn-mdp' ]
        then
            for lambda in $lambda
            do
                python compute_dim.py --batch_size 1 --mask_c $lambda --algo gnn-mdp --flag $flag --lr 0.01 --epochs 1000 --num_iters 1
            done            
        else
            python compute_dim.py --flag $flag --algo algo2
        fi
    done
done
