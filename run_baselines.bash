#!/bin/bash

algos='gnn-mdp'
lambdas='3.'
flags='2 3 4 5 6'
for flag in $flags
do
    for lambda in $lambdas
    do
        echo $lambda
        python compute_dim.py --algo gnn-mdp --flag $flag --gnn_model distmask --suffix distmask$1 --mask_c $lambda --num_iters 1 --do_time --suffix time
    done
done


