#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --time=1-00:00
#SBATCH --mem-per-cpu=5000M
#SBATCH --array=0-2

module load scipy-stack
source /project/6005607/ahollid/ptlearn/bin/activate
PYTHONPATH="/home/ahollid/transit_learning:/home/ahollid/ptsim:$PYTHONPATH"

datasets=(
    "20_nodes/voronoi_test"
    "50_nodes/voronoi_test"
    "100_nodes/voronoi_test"
)
n_routes_by_dataset=(
    4
    10
    20
)
# assuming 30s * 20 for python = 600s per node
times_by_dataset=(
    12000
    30000
    60000
)
names=(
    "graph_20"
    "graph_50"
    "graph_100"
)

dataset_dir=/home/ahollid/scratch/datasets/
dd="${datasets[$SLURM_ARRAY_TASK_ID]}"
ddname="${names[$SLURM_ARRAY_TASK_ID]}"
output_csv="/home/ahollid/scratch/${ddname}_hh.csv"

dataset="$dataset_dir/$dd"
nroutes="${n_routes_by_dataset[$SLURM_ARRAY_TASK_ID]}"
runtime="${times_by_dataset[$SLURM_ARRAY_TASK_ID]}"
python /home/ahollid/transit_learning/learning/hyperheuristics.py \
    --csv $dataset $nroutes -s $runtime --cpu --onlyfirst --bs 1 >> $output_csv

