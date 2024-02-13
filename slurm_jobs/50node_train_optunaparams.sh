#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --time=1-00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=15000M

# module load python/3.8
module load scipy-stack
module load mariadb
source /project/6005607/ahollid/ptlearn/bin/activate
PYTHONPATH="/home/ahollid/transit_learning:/home/ahollid/ptsim:$PYTHONPATH"
dataset="mixed"
python /home/ahollid/transit_learning/learning/inductive_route_learning.py \
    "/home/ahollid/scratch/datasets/50_nodes/$dataset" \
    --cfg /home/ahollid/transit_learning/cfg/optuna_pruning_cfg.yaml \
    --logdir /home/ahollid/scratch/tb_logs -o /home/ahollid/scratch/outputs \
    --rn 50node_optunaParams --ne 20
