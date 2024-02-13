#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --time=11-00:00
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=20000M
#SBATCH --array=1-20

# module load python/3.8
module load scipy-stack
module load mariadb
source /project/6005607/ahollid/ptlearn/bin/activate
PYTHONPATH="/home/ahollid/transit_learning:/home/ahollid/ptsim:$PYTHONPATH"
dataset="mixed"
python /home/ahollid/transit_learning/learning/tune_hyperparams.py \
    "/home/ahollid/scratch/datasets/50_nodes/$dataset" \
    50nodes_fullmixed_study --ne 10 -t 260 --bs 16
