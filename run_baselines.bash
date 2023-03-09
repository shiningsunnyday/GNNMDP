#!/bin/bash

algos='gnn-mdp'
lr=0.01
lambdas='0.03 0.1 0.3 1. 3. 10.0'
for lambda in $lambdas
do
    echo $1
    echo $lambda
    python compute_dim.py --algo gnn-mdp --flag $1 --gnn_model distmask --suffix distmask$1 --mask_c $lambda
done


