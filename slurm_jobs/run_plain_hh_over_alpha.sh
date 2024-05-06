#!/bin/bash
#SBATCH --account=rrg-dpmeger
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=5000M
#SBATCH --array=0-109

# 1.5 + 2 + (20min+4) + (8+8) + (8+12) = 43.84 hours, so 48 to be safe

date

# skip 1s and 0s, as they're effectively already done (PP and OP, respectively)
alphas=(
    0.0
    0.1
    0.2
    0.3
    0.4
    0.5
    0.6
    0.7
    0.8
    0.9
    1.0
)

# comment out for local running
module load scipy-stack
module load mariadb
source /project/6005607/ahollid/ptlearn/bin/activate
PYTHONPATH="/home/ahollid/transit_learning:/home/ahollid/ptsim:$PYTHONPATH"
out_dir=/home/ahollid/scratch/results/
dataset_dir=/home/ahollid/scratch/datasets/mumford_dataset/Instances
weights_dir=/home/ahollid/scratch/weights/seeds

# uncomment for local running
# out_dir=output_csvs
# dataset_dir=datasets/mumford_dataset/Instances
# weights_dir=output
# SLURM_TMPDIR='.'
# SLURM_ARRAY_TASK_ID=0

seed=$((SLURM_ARRAY_TASK_ID / 11))
model=$weights_dir/inductive_seed_$seed.pt

alpha_idx=$((SLURM_ARRAY_TASK_ID % 11))
alpha=${alphas[$alpha_idx]}
# beta = 1.0 - alpha
beta=$(echo "1.0-$alpha" | bc)

cities=(
    mandl
    mumford0
    mumford1
    mumford2
    mumford3
)

for city in "${cities[@]}"
do
    base_run_name=${city}_s${seed}_a${alpha}
    
    # plain HH init
    python learning/hyperheuristics.py hydra/job_logging=disabled +eval=$city \
        eval.dataset.path=$dataset_dir  n_iterations=500000 \
        experiment.cost_function.kwargs.demand_time_weight=$alpha \
        experiment.cost_function.kwargs.route_time_weight=$beta \
        experiment.seed=$seed +run_name=plain_$base_run_name \
        experiment.logdir=$SLURM_TMPDIR/tb_logs \
        >> $out_dir/hh_plain_pareto_$alpha.csv
done
cp -r $SLURM_TMPDIR/tb_logs/* /home/ahollid/scratch/tb_logs/

date
