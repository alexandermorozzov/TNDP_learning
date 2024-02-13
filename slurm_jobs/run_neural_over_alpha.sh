#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=5000M
#SBATCH --array=0-9

# skip 1s and 0s, as they're effectively already done (PP and OP, respectively)
alphas=(
    0.1
    0.2
    0.3
    0.4
    0.6
    0.7
    0.8
    0.9
)
betas=(
    0.9
    0.8
    0.7
    0.6
    0.4
    0.3
    0.2
    0.1
)

module load scipy-stack
module load mariadb
source /project/6005607/ahollid/ptlearn/bin/activate
PYTHONPATH="/home/ahollid/transit_learning:/home/ahollid/ptsim:$PYTHONPATH"

out_dir=/home/ahollid/scratch/pareto/
dataset_dir=/home/ahollid/scratch/datasets/mumford_dataset/Instances

seed=$SLURM_ARRAY_TASK_ID

for ((ii=0; ii<${#alphas[@]}; ii++))
do
    alpha=${alphas[$ii]}
    beta=${betas[$ii]}
    bash scripts/mumford_eval.sh learning/eval_route_generator.py \
        eval_model_mumford hydra/job_logging=disabled \
        eval.dataset.path=$dataset_dir \
        +model.weights=/home/ahollid/scratch/weights/inductive_seed_$seed.pt \
        experiment.cost_function.kwargs.demand_time_weight=$alpha \
        experiment.cost_function.kwargs.route_time_weight=$beta \
        experiment.seed=$seed +run_name=alpha_$alpha \
        >> $out_dir/neural_pareto_$alpha.csv
done

cp $SLURM_TMPDIR/tb_logs/* /home/ahollid/scratch/tb_logs/