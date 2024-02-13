#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --time=6-00:00
#SBATCH --mem-per-cpu=5000M
#SBATCH --array=0-2

module load scipy-stack
source /project/6005607/ahollid/ptlearn/bin/activate
PYTHONPATH="/home/ahollid/transit_learning:/home/ahollid/ptsim:$PYTHONPATH"

datasets=(
    "graph_250"
    "graph_500"
    "graph_750"
#    "graph_1000"
)
n_routes_by_dataset=(
    35
    65
    95
#    125
)
# assuming 30s * 20 for python = 600s per node
times_by_dataset=(
    150000
    300000
    450000
#    600000
)

dataset_dir=/home/ahollid/scratch/datasets/scaling_graphs
dsdir_name=$(basename $dataset_dir)
dd="${datasets[$SLURM_ARRAY_TASK_ID]}"
output_csv="/home/ahollid/scratch/${dsdir_name}_${dd}_hh.csv"

dataset="$dataset_dir/$dd"
nroutes="${n_routes_by_dataset[$SLURM_ARRAY_TASK_ID]}"
runtime="${times_by_dataset[$SLURM_ARRAY_TASK_ID]}"
python /home/ahollid/transit_learning/learning/hyperheuristics.py \
    --csv $dataset $nroutes -s $runtime --cpu --bs 1 >> $output_csv

