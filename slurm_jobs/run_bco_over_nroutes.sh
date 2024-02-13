module load scipy-stack
source /project/6005607/ahollid/ptlearn/bin/activate
PYTHONPATH="/home/ahollid/transit_learning:/home/ahollid/ptsim:$PYTHONPATH"

datasets=(
    "4grid_test"
    "8grid_test"
    "inknn_test"
    "voronoi_test"
)

dataset_dir=$1
cfg=$2
dsdir_name=$(basename $dataset_dir)
dd="${datasets[$SLURM_ARRAY_TASK_ID]}"
output_csv="/home/ahollid/scratch/${dsdir_name}_${dd}_bco.csv"

echo "$dd,,," >> $output_csv
dataset="$dataset_dir/$dd"
python /home/ahollid/transit_learning/learning/bee_colony.py \
    --config-name $cfg dataset=$dataset logdir=~/scratch/tb_logs \
    hydra/job_logging=disabled >> $output_csv
