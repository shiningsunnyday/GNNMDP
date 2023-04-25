#!/bin/bash

flags='2 3 4 5 6'
lrs='0.003'

for flag in $flags
do
    for lr in $lrs
    do
        python compute_dim.py --flag $flag --algo algo2 --lr $lr --do_time --suffix time
    done  
done


