#!/bin/bash

algos='algo2'
gnns=$1
lr=0.003
for gnn in $gnns
do
    echo $gnn
    python compute_dim.py --algo algo2 --flag $2 --lr $lr --gnn_model $gnn --suffix algo2$gnn
done

