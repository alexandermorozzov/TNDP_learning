#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=5000M
#SBATCH --array=0-9

module load python/3.8
# module load scipy-stack
module load mariadb
source /project/6005607/ahollid/ptlearn/bin/activate
PYTHONPATH="/home/ahollid/transit_learning:/home/ahollid/ptsim:$PYTHONPATH"

# eg. online_bco, online_neural, online_nbco
config_file=$1
# eg. mine, pp, op
cost_function=$2
# a time in seconds
timeper=$3

seed=$SLURM_ARRAY_TASK_ID
# right now this is hardcoded to mumford3.  Could make it a parameter later.
city=mumford3
dataset_dir=/home/ahollid/scratch/datasets/mumford_dataset/Instances
out_dir=/home/ahollid/scratch/online_over_seeds
weights=/home/ahollid/scratch/weights/seeds/inductive_seed_${seed}.pt

# assemble the name of the output csv file
out_file=$out_dir/${config_file}_${cost_function}_${timeper}.csv
# run the experiment and write the results to a CSV file
python learning/online_replanning.py online_planner=$config_file +eval=$city \
    iter_time_limit_s=$timeper experiment/cost_function=$cost_function \
    eval.dataset.path=$dataset_dir experiment.seed=$seed \
    +model.weights=$weights hydra/job_logging=disabled >> $out_file
