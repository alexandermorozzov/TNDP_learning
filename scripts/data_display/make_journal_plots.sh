python scripts/data_display/plot_pareto.py \
    result_csvs/gae/s100_pareto_*.csv result_csvs/gae/bco_pareto_* \
    result_csvs/gae/neural_bco_pareto_* result_csvs/gae/s40k_pareto_* \
    --labels -e Mumford1 -e Mumford2 -e Mumford3 -o ~/Desktop/figs/40k.pdf

python scripts/data_display/plot_pareto.py \
    result_csvs/gae/neural_bco_no2_pareto_* \
    result_csvs/gae/neural_bco_pareto_* \
    --labels -e Mumford1 -e Mumford2 -e Mumford3 -o ~/Desktop/figs/no2.pdf 

python scripts/data_display/plot_pareto.py \
    result_csvs/gae/neural_bco_pareto_* \
    result_csvs/post_simfix/neural_bco_random_pareto_* --labels \
    --nc NEA --nc "RC-EA" -e Mumford1 -e Mumford2 -e Mumford3 \
    -o ~/Desktop/figs/rcea.pdf

python scripts/data_display/plot_pareto.py \
    result_csvs/post_simfix/stl_laval_1.0.csv \
    result_csvs/gae/s100_laval_* \
    result_csvs/gae/neural_bco_laval_* \
    result_csvs/gae/neural_bco_random_laval_1.* --labels --fs 2 --asymmetric \
    -o ~/Desktop/figs/laval_pareto.pdf
