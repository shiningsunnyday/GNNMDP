#!/bin/bash

algos='gnn-mdp'
lr=0.01
lambdas='1.'
flags='2 3 4 5 6'
for flag in $flags
do
    for lambda in $lambdas
    do
        echo $flag
        echo $lambda
        python compute_dim.py --epochs 2 --algo gnn-mdp --flag $flag --gnn_model distmask --suffix omptime --distmask_c $lambda --mask_c 0 --do_omp True --batch_size 1 --num_iters 1
    done
done

