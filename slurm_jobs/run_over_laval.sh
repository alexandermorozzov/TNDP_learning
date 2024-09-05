#!/bin/bash
#SBATCH --account=rrg-dpmeger
#SBATCH --time=40:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=5000M
#SBATCH --array=0-29

date

# skip 1s and 0s, as they're effectively already done (PP and OP, respectively)
alphas=(
    0.0
    # 0.1
    # 0.2
    # 0.3
    # 0.4
    0.5
    # 0.6
    # 0.7
    # 0.8
    # 0.9
    1.0
)

# comment out for local running
module load scipy-stack
module load mariadb
source /project/6005607/ahollid/ptlearn/bin/activate
PYTHONPATH="/home/ahollid/transit_learning:/home/ahollid/ptsim:$PYTHONPATH"
out_dir=/home/ahollid/scratch/results/
dataset_dir=/home/ahollid/scratch/datasets/laval
weights_dir=/home/ahollid/scratch/weights/seeds

# uncomment for local running
# out_dir=output_csvs
# dataset_dir=datasets/laval
# weights_dir=output
# SLURM_TMPDIR='.'
# SLURM_ARRAY_TASK_ID=0

seed=$((SLURM_ARRAY_TASK_ID / 3))
model=$weights_dir/inductive_seed_$seed.pt
# model=output/inductive_seed_$seed.pt

alpha_idx=$((SLURM_ARRAY_TASK_ID % 3))
alpha=${alphas[$alpha_idx]}
# beta = 1.0 - alpha
beta=$(echo "1.0-$alpha" | bc)

base_run_name=laval_s${seed}_a${alpha}

# run the network in greedy mode
python learning/eval_route_generator.py hydra/job_logging=disabled \
    eval.dataset.path=$dataset_dir +eval=lavalDA +model.weights=$model \
    experiment.cost_function.kwargs.demand_time_weight=$alpha \
    experiment.cost_function.kwargs.route_time_weight=$beta \
    experiment.seed=$seed +run_name=greedy_$base_run_name n_samples=null \
    model.route_generator.kwargs.force_linking_unlinked=true \
    >> $out_dir/greedy_laval_$alpha.csv
# and with 100 samples
python learning/eval_route_generator.py hydra/job_logging=disabled \
    eval.dataset.path=$dataset_dir +eval=lavalDA +model.weights=$model \
    experiment.cost_function.kwargs.demand_time_weight=$alpha \
    experiment.cost_function.kwargs.route_time_weight=$beta \
    experiment.seed=$seed +run_name=s100_$base_run_name n_samples=100 \
    model.route_generator.kwargs.force_linking_unlinked=true \
    batch_size=32 >> $out_dir/s100_laval_$alpha.csv

init_soln_file=output_routes/nn_construction_s100_${base_run_name}_routes.pkl

python learning/bee_colony.py --config-name bco_mumford \
    hydra/job_logging=disabled eval.dataset.path=$dataset_dir +eval=lavalDA \
    experiment.cost_function.kwargs.demand_time_weight=$alpha \
    experiment.cost_function.kwargs.route_time_weight=$beta \
    experiment.seed=$seed +run_name=ea_$base_run_name \
    experiment.logdir=$SLURM_TMPDIR/tb_logs force_linking_unlinked=true \
    +init_solution_file=$init_soln_file >> $out_dir/bco_laval_$alpha.csv
python learning/bee_colony.py --config-name neural_bco_mumford \
    hydra/job_logging=disabled eval.dataset.path=$dataset_dir +eval=lavalDA \
    experiment.cost_function.kwargs.demand_time_weight=$alpha \
    experiment.cost_function.kwargs.route_time_weight=$beta \
    experiment.seed=$seed +run_name=nea_$base_run_name +model.weights=$model \
    experiment.logdir=$SLURM_TMPDIR/tb_logs force_linking_unlinked=true \
    +init_solution_file=$init_soln_file >> $out_dir/neural_bco_laval_$alpha.csv

# random "NEA"
# first use it to generate a starting solution...
python learning/eval_route_generator.py hydra/job_logging=disabled \
    eval.dataset.path=$dataset_dir +eval=lavalDA model=random_path_combiner \
    experiment.cost_function.kwargs.demand_time_weight=$alpha \
    experiment.cost_function.kwargs.route_time_weight=$beta \
    experiment.seed=$seed +run_name=s100_random_$base_run_name n_samples=100 \
    batch_size=32 model.route_generator.kwargs.force_linking_unlinked=true \
# ...then use that as the initial solution for the NEA
init_soln_file=output_routes/nn_construction_s100_random_${base_run_name}_routes.pkl
python learning/bee_colony.py --config-name=neural_bco_mumford \
    hydra/job_logging=disabled +eval=lavalDA eval.dataset.path=$dataset_dir \
    experiment.cost_function.kwargs.demand_time_weight=$alpha \
    experiment.cost_function.kwargs.route_time_weight=$beta \
    experiment.seed=$seed +run_name=random_$base_run_name \
    experiment.logdir=$SLURM_TMPDIR/tb_logs model=random_path_combiner \
    +init_solution_file=$init_soln_file force_linking_unlinked=true \
    >> $out_dir/neural_bco_random_laval_$alpha.csv

cp -r $SLURM_TMPDIR/tb_logs/* /home/ahollid/scratch/tb_logs/

date
