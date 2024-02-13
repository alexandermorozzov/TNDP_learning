#!/bin/bash
#SBATCH --account=rrg-dpmeger
#SBATCH --time=1:15:00
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=5000M
#SBATCH --array=0-9

module load python/3.8
module load mariadb
source /project/6005607/ahollid/ptlearn/bin/activate
PYTHONPATH="/home/ahollid/transit_learning:/home/ahollid/ptsim:$PYTHONPATH"

seed=$SLURM_ARRAY_TASK_ID

python learning/inductive_route_learning.py --config-name bestsofar_20nodes \
       dataset.kwargs.path=/home/ahollid/scratch/datasets/20_nodes/mixed \
       +run_name=seed_$seed experiment.seed=$seed \
       +outdir=/home/ahollid/scratch/weights/seeds
