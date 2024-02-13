#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --time=1-00:00
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=5000M
#SBATCH --array=0-4

dataset_dir=/home/ahollid/scratch/datasets/100_nodes
cfg=bco_100nodes
bash /home/ahollid/transit_learning/slurm_jobs/run_bco_over_nroutes.sh \
    $dataset_dir $cfg