#!/bin/bash
#SBATCH --account=rrg-dpmeger
#SBATCH --time=30:00
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=5000M
#SBATCH --array=0-109

date

# skip 1s, 0.5s, and 0s, as they're effectively already done (PP and OP, 
# respectively)
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
    # LC greedy
    python learning/eval_route_generator.py \
        +eval=$city hydra/job_logging=disabled eval.dataset.path=$dataset_dir \
        +model.weights=$model \
        experiment.cost_function.kwargs.demand_time_weight=$alpha \
        experiment.cost_function.kwargs.route_time_weight=$beta \
        experiment.seed=$seed n_samples=null \
        +run_name=greedy_${base_run_name} \
        >> $out_dir/greedy_pareto_$alpha.csv
    # LC-100
    python learning/eval_route_generator.py \
        +eval=$city hydra/job_logging=disabled eval.dataset.path=$dataset_dir \
        +model.weights=$model \
        experiment.cost_function.kwargs.demand_time_weight=$alpha \
        experiment.cost_function.kwargs.route_time_weight=$beta \
        experiment.seed=$seed n_samples=100 \
        +run_name=s100_${base_run_name} \
        >> $out_dir/s100_pareto_$alpha.csv
    
    # first use it to generate a starting solution (this is RC-100)...
    python learning/eval_route_generator.py \
        +eval=$city hydra/job_logging=disabled eval.dataset.path=$dataset_dir \
        model=random_path_combiner experiment.seed=$seed n_samples=100 \
        experiment.cost_function.kwargs.demand_time_weight=$alpha \
        experiment.cost_function.kwargs.route_time_weight=$beta \
        +run_name=s100_random_${base_run_name} \
        >> $out_dir/random_constructor_pareto_$alpha.csv
    # also try alpha-halt
    python learning/eval_route_generator.py \
        +eval=$city hydra/job_logging=disabled eval.dataset.path=$dataset_dir \
        model=random_path_combiner experiment.seed=$seed n_samples=100 \
        experiment.cost_function.kwargs.demand_time_weight=$alpha \
        experiment.cost_function.kwargs.route_time_weight=$beta \
        route_generator.kwargs.halt_prob_is_route_time_weight=true \
        +run_name=s100_random_${base_run_name} \
        >> $out_dir/random_constructor_alphahalt_pareto_$alpha.csv

    # LC-40k
    python learning/eval_route_generator.py \
        +eval=$city hydra/job_logging=disabled eval.dataset.path=$dataset_dir \
        +model.weights=$model \
        experiment.cost_function.kwargs.demand_time_weight=$alpha \
        experiment.cost_function.kwargs.route_time_weight=$beta \
        experiment.seed=$seed n_samples=40000 \
        +run_name=s40k_${base_run_name} \
        >> $out_dir/s40k_pareto_$alpha.csv
done
cp -r $SLURM_TMPDIR/tb_logs/* /home/ahollid/scratch/tb_logs/

date
