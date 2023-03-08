#!/bin/bash

algos='gnn-mdp'
flags='1 2 3 4 5 6'
gnns=$1
lr=0.01
for gnn in $gnns
do
    for flag in $flags
    do
	echo $gnn
	python compute_dim.py --batch_size 1 --mask_c 3.0 --algo gnn-mdp --flag $flag --lr $lr --epochs 1000 --num_iters 1 --gnn_model $gnn --suffix $gnn
    done
done

