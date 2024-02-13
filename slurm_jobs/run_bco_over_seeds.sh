#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=3000M
#SBATCH --array=0-9

module load scipy-stack
module load mariadb
source /project/6005607/ahollid/ptlearn/bin/activate
PYTHONPATH="/home/ahollid/transit_learning:/home/ahollid/ptsim:$PYTHONPATH"

dataset_dir=/home/ahollid/scratch/datasets/mumford_dataset/Instances
out_dir=/home/ahollid/scratch/results/

seed=$SLURM_ARRAY_TASK_ID

bash scripts/mumford_eval.sh learning/bee_colony.py bco_mumford \
    hydra/job_logging=disabled experiment/cost_function=mine \
    eval.dataset.path=$dataset_dir experiment.seed=$seed \
    +run_name=balanced_seed_$seed experiment.logdir=$SLURM_TMPDIR/tb_logs \
    >> $out_dir/bco_seeds_balanced.csv
bash scripts/mumford_eval.sh learning/bee_colony.py bco_mumford \
    hydra/job_logging=disabled experiment/cost_function=mine_pp \
    eval.dataset.path=$dataset_dir experiment.seed=$seed \
    +run_name=pp_seed_$seed experiment.logdir=$SLURM_TMPDIR/tb_logs \
    >> $out_dir/bco_seeds_pp.csv
bash scripts/mumford_eval.sh learning/bee_colony.py bco_mumford \
    hydra/job_logging=disabled experiment/cost_function=mine_op \
    eval.dataset.path=$dataset_dir experiment.seed=$seed \
    +run_name=op_seed_$seed experiment.logdir=$SLURM_TMPDIR/tb_logs \
    >> $out_dir/bco_seeds_op.csv

cp $SLURM_TMPDIR/tb_logs/* /home/ahollid/scratch/tb_logs/