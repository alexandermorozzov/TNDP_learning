#!/bin/bash

for config_file in "bco" "neural" "neural_bco"
do
    for cost_fn in "mine" "pp" "op"
    do
        for timeper in 30 60 120
        do
            sbatch slurm_jobs/online_over_seeds.sh $config_file $cost_fn $timeper
        done
    done
done