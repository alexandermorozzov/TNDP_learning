#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --time=2:30:00
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=5000M
#SBATCH  --array=0-24

module load scipy-stack
source /project/6005607/ahollid/ptlearn/bin/activate
PYTHONPATH="/home/ahollid/transit_learning:/home/ahollid/ptsim:$PYTHONPATH"

datasets=(
    "inknn_test"
    "outknn_test"
    "voronoi_test"
    "4grid_test"
    "8grid_test"
)
nroutes=(
    1
    2
    4
    6
    8
)
dataset_dir=/home/ahollid/scratch/datasets/20_nodes
dsdir_name=$(basename $dataset_dir)
# nine n_routes per dataset
dataset_idx=$(expr $SLURM_ARRAY_TASK_ID / 5)
dd="${datasets[$dataset_idx]}"
output_csv="/home/ahollid/scratch/hh/${dsdir_name}_${dd}_hh.csv"
nroutes_idx=$(expr $SLURM_ARRAY_TASK_ID % 5)
nroutes=${nroutes[$nroutes_idx]}
dataset="$dataset_dir/$dd"
python /home/ahollid/transit_learning/learning/hyperheuristics.py --csv \
    $dataset $nroutes -s  --bs 1024 --logdir $SLURM_TMPDIR/tb_logs >> $output_csv
cp $SLURM_TMPDIR/tb_logs/* ~/scratch/tb_logs/