#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --time=2:10:00
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=5000M
#SBATCH --array=0-9

module load scipy-stack
module load mariadb
source /project/6005607/ahollid/ptlearn/bin/activate
PYTHONPATH="/home/ahollid/transit_learning:/home/ahollid/ptsim:$PYTHONPATH"

dataset_dir=/home/ahollid/scratch/datasets/mumford_dataset/Instances
out_dir=/home/ahollid/scratch/

seed=$SLURM_ARRAY_TASK_ID

bash scripts/mumford_eval.sh learning/bee_colony.py neural_bco_mumford \
    hydra/job_logging=disabled experiment/cost_function=mine \
    eval.dataset.path=$dataset_dir \
    +model.weights=/home/ahollid/scratch/weights/inductive_seed_${seed}.pt \
    experiment.seed=$seed +run_name=at1b_balanced_seed_$seed \
    experiment.logdir=$SLURM_TMPDIR/tb_logs +n_type1_bees=10 \
    >> $out_dir/neural_bco_at1b_seeds_balanced.csv
bash scripts/mumford_eval.sh learning/bee_colony.py neural_bco_mumford \
    hydra/job_logging=disabled experiment/cost_function=mine_pp \
    eval.dataset.path=$dataset_dir \
    +model.weights=/home/ahollid/scratch/weights/inductive_seed_${seed}.pt \
    experiment.seed=$seed +run_name=at1b_pp_seed_$seed \
    experiment.logdir=$SLURM_TMPDIR/tb_logs +n_type1_bees=10 \
    >> $out_dir/neural_bco_at1b_seeds_pp.csv
bash scripts/mumford_eval.sh learning/bee_colony.py neural_bco_mumford \
    hydra/job_logging=disabled experiment/cost_function=mine_op \
    eval.dataset.path=$dataset_dir \
    +model.weights=/home/ahollid/scratch/weights/inductive_seed_${seed}.pt \
    experiment.seed=$seed +run_name=at1b_op_seed_$seed \
    experiment.logdir=$SLURM_TMPDIR/tb_logs +n_type1_bees=10 \
    >> $out_dir/neural_bco_at1b_seeds_op.csv


cp $SLURM_TMPDIR/tb_logs/* /home/ahollid/scratch/tb_logs/