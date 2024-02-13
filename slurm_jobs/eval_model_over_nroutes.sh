module load scipy-stack
source /project/6005607/ahollid/ptlearn/bin/activate
PYTHONPATH="/home/ahollid/transit_learning:/home/ahollid/ptsim:$PYTHONPATH"

weights=$1
dataset_dir=$2
cfg=$3
extra_args=("${@:4}")
weights_name=$(basename -s .pt $weights)
datasets=(
    "4grid_test"
    "8grid_test"
    "inknn_test"
    "voronoi_test"
)
# dataset_dir=/home/ahollid/scratch/datasets/50_nodes
dsdir_name=$(basename $dataset_dir)
output_csv="/home/ahollid/scratch/${dsdir_name}_${weights_name}.csv"

for dd in ${datasets[@]}
do
    dataset="$dataset_dir/$dd"
    echo "$dd,,,,," >> $output_csv
	python /home/ahollid/transit_learning/learning/eval_route_generator.py \
	    --config-name $cfg +model.weights=$weights +dataset=$dataset \
        hydra/job_logging=disabled >> $output_csv
done
