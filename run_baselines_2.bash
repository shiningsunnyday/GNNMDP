#!/bin/bash

algos='gnn-mdp'
lr=0.01
lambdas='0.01 0.03 0.1 0.3 1. 3.'
for lambda in $lambdas
do
    echo $1
    echo $lambda
    python compute_dim.py --epochs 2 --algo gnn-mdp --flag $1 --gnn_model distmask --suffix distmask --distmask_c $lambda --mask_c 0 --do_omp True --batch_size 1 --num_iters 1
done


