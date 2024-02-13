#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --time=5-00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=40000M

# module load python/3.8
module load scipy-stack
module load mariadb
source /project/6005607/ahollid/ptlearn/bin/activate
PYTHONPATH="/home/ahollid/transit_learning:/home/ahollid/ptsim:$PYTHONPATH"
dataset="mixed"
python /home/ahollid/transit_learning/learning/inductive_route_learning.py \
    "/home/ahollid/scratch/datasets/100_nodes/$dataset" \
    --cfg /home/ahollid/transit_learning/cfg/optuna_pruning_faster.yaml \
    --logdir /home/ahollid/scratch/tb_logs -o /home/ahollid/scratch/outputs \
    --rn optuna_params --ne 10
