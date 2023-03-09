#!/bin/bash

algos='algo2'
flags='1 2 3 4 5 6'
gnns=$1
lr=0.003
for gnn in $gnns
do
    for flag in $flags
    do
        echo $gnn
        python compute_dim.py --algo algo2 --flag $flag --lr $lr --gnn_model $gnn --suffix algo2$gnn
    done
done

