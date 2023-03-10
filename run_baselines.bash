#!/bin/bash

algos='gnn-mdp'
lambdas='3. 10.0'
for lambda in $lambdas
do
    echo $1
    echo $lambda
    python compute_dim.py --algo gnn-mdp --flag $1 --gnn_model distmask --suffix distmask$1 --mask_c $lambda --num_iters 1
done


