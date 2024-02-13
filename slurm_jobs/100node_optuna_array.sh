#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --time=1-12:00
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=5000M
#SBATCH --array=1-20

# module load python/3.8
module load scipy-stack
module load mariadb
source /project/6005607/ahollid/ptlearn/bin/activate
PYTHONPATH="/home/ahollid/transit_learning:/home/ahollid/ptsim:$PYTHONPATH"
dataset="100nodes_mixed_mini"
python /home/ahollid/transit_learning/learning/tune_hyperparams.py \
    "/home/ahollid/scratch/$dataset" \
    100nodes_mixed_study --ne 10 -t 300 --mysql
