#!/bin/bash
#SBATCH --account=rrg-dpmeger
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --array=0-9

module load python/3.8
module load mariadb
source /project/6005607/ahollid/ptlearn/bin/activate
PYTHONPATH="/home/ahollid/transit_learning:/home/ahollid/ptsim:$PYTHONPATH"

seed=$SLURM_ARRAY_TASK_ID

python learning/inductive_route_learning.py --config-name bestsofar_100nodes \
       dataset.kwargs.path=/home/ahollid/scratch/datasets/100_nodes/mixed \
       +run_name=seed_100nodes_$seed experiment.seed=$seed \
       experiment.logdir=$SLURM_TMPDIR/tb_logs \
       +outdir=/home/ahollid/scratch/weights/seeds

mkdir -p /home/ahollid/scratch/tb_logs/
cp -r $SLURM_TMPDIR/tb_logs/* /home/ahollid/scratch/tb_logs/
