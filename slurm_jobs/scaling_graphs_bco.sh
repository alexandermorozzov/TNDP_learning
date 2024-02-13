#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --time=05:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=5G
#SBATCH --array=0-6

module load scipy-stack
source /project/6005607/ahollid/ptlearn/bin/activate
PYTHONPATH="/home/ahollid/transit_learning:/home/ahollid/ptsim:$PYTHONPATH"

datasets=(
    "graph_20"
    "graph_50"
    "graph_100"
    "graph_250"
    "graph_500"
    "graph_750"
    "graph_1000"
)
n_routes_by_dataset=(
    4
    8
    15
    35
    65
    95
    125
)

dataset_dir=/home/ahollid/scratch/datasets/scaling_graphs
dsdir_name=$(basename $dataset_dir)
dd="${datasets[$SLURM_ARRAY_TASK_ID]}"
output_csv="/home/ahollid/scratch/${dsdir_name}_${dd}_bco.csv"

# echo "$dd,,," >> $output_csv
dataset="$dataset_dir/$dd"
nroutes="${n_routes_by_dataset[$SLURM_ARRAY_TASK_ID]}"
python /home/ahollid/transit_learning/learning/bee_colony.py --csv --bbs 5 \
    $dataset $nroutes --bs 1 --logdir $SLURM_TMPDIR/tb_logs >> $output_csv
