#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --time=0:10:00
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=5000M

# module load python/3.8
module load scipy-stack
source /project/6005607/ahollid/ptlearn/bin/activate
PYTHONPATH="/home/ahollid/transit_learning:/home/ahollid/ptsim:$PYTHONPATH"
didataset="20nodes_mixed_mini"
python /home/ahollid/transit_learning/learning/tune_hyperparams.py \
    "/home/ahollid/scratch/$dataset" \
    20nodes_mixed_study --ne 10 -t 5

