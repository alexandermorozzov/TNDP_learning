#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=5000M
#SBATCH --array=0-19

date

# skip 1s and 0s, as they're effectively already done (PP and OP, respectively)
alphas=(
    0.0
    1.0
)
betas=(
    1.0
    0.0
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

n_alphas=${#alphas[@]}

seed=$((SLURM_ARRAY_TASK_ID / n_alphas))
model=$weights_dir/inductive_seed_$seed.pt

alpha_idx=$((SLURM_ARRAY_TASK_ID % n_alphas))
alpha=${alphas[$alpha_idx]}
beta=${betas[$alpha_idx]}

bash scripts/mumford_eval.sh learning/bee_colony.py neural_bco_mumford \
    hydra/job_logging=disabled eval.dataset.path=$dataset_dir \
    +model.weights=$model \
    experiment.cost_function.kwargs.demand_time_weight=$alpha \
    experiment.cost_function.kwargs.route_time_weight=$beta \
    experiment.seed=$seed +run_name=alpha_$alpha \
    experiment.logdir=$SLURM_TMPDIR/tb_logs \
    >> $out_dir/tf_neural_bco_pareto_$alpha.csv

cp -r $SLURM_TMPDIR/tb_logs/* /home/ahollid/scratch/tb_logs/

date
