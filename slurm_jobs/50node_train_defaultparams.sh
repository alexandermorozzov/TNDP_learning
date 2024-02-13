#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --time=1-12:00
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=10000M

# module load python/3.8
module load scipy-stack
module load mariadb
source /project/6005607/ahollid/ptlearn/bin/activate
PYTHONPATH="/home/ahollid/transit_learning:/home/ahollid/ptsim:$PYTHONPATH"
dataset="50nodes_mixed"
python /home/ahollid/transit_learning/learning/inductive_route_learning.py \
    "/home/ahollid/scratch/$dataset" \
    --logdir /home/ahollid/scratch/tb_logs -o /home/ahollid/scratch/outputs \
    --rn default_params
