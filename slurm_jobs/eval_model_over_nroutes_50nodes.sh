#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --time=09:00:00
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=5000M

weights=$1
extra_args=("${@:2}")
dataset_dir=/home/ahollid/scratch/datasets/50_nodes
cfg=eval_model_50nodes

bash /home/ahollid/transit_learning/slurm_jobs/eval_model_over_n_routes.sh \
    $weights $dataset_dir $cfg $extra_args