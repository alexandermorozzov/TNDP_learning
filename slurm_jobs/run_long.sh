#!/bin/bash
#SBATCH --account=rrg-dpmeger
#SBATCH --time=40:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=10G
#SBATCH --array=8

date

# skip 1s and 0s, as they're effectively already done (PP and OP, respectively)
alphas=(
    0.0
    0.5
    1.0
)

# comment out for local running
module load scipy-stack
module load mariadb
source /project/6005607/ahollid/ptlearn/bin/activate
PYTHONPATH="/home/ahollid/transit_learning:/home/ahollid/ptsim:$PYTHONPATH"
out_dir=/home/ahollid/scratch/results
dataset_dir=/home/ahollid/scratch/datasets/mumford_dataset/Instances
weights_dir=/home/ahollid/scratch/weights/seeds

# uncomment for local running
# out_dir=output_csvs
# dataset_dir=datasets/mumford_dataset/Instances
# weights_dir=output
# SLURM_TMPDIR='.'
# SLURM_ARRAY_TASK_ID=0

seed=$((SLURM_ARRAY_TASK_ID / 3))
model=$weights_dir/inductive_seed_$seed.pt

alpha_idx=$((SLURM_ARRAY_TASK_ID % 3))
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
    # use the 100 sample solution as the initial solution for the EA
    init_soln_file=output_routes/nn_construction_s100_${base_run_name}_routes.pkl
    python learning/bee_colony.py --config-name=neural_bco_mumford \
        hydra/job_logging=disabled +eval=$city eval.dataset.path=$dataset_dir \
        +model.weights=$model \
        experiment.cost_function.kwargs.demand_time_weight=$alpha \
        experiment.cost_function.kwargs.route_time_weight=$beta \
        experiment.seed=$seed +run_name=long_$base_run_name \
        experiment.logdir=$SLURM_TMPDIR/tb_logs \
        +init_solution_file=$init_soln_file n_iterations=4000 \
        >> $out_dir/neural_bco_long_$alpha.csv
done
cp -r $SLURM_TMPDIR/tb_logs/* /home/ahollid/scratch/tb_logs/

date
