fs="2.75"

# init experiments (EA and HH)
for prefix in ea hh; do
    python scripts/data_display/hypervolume.py \
        result_csvs/init_experiments/init_${prefix}_plain* \
        result_csvs/init_experiments/init_${prefix}_s100* \
        --nc XYZ --nc 'LC-100 (ours)' --nc NEA --nc 'Nikolić (2013)' \
        --nc 'Ahmed (2019)' --fs $fs -o ~/Desktop/figs/${prefix}_hv_bars.pdf
done

# LC-100 vs Random pareto plot
python scripts/data_display/plot_pareto.py \
    result_csvs/ppo/s100_pareto_* \
    result_csvs/march26results/random_constructor_pareto_* --env Mumford3 \
    --nc 'LC-100 (ours)' --nc EA --nc NEA --nc 'RC-100' \
    --labels -o ~/Desktop/figs/lc100_results/pareto.pdf
    
# LC-100 vs random
python scripts/data_display/hypervolume.py \
    result_csvs/ppo/s100_pareto_* \
    result_csvs/march26results/random_constructor_pareto_* \
    --nc 'LC-100 (ours)' --nc EA --nc NEA --nc 'RC-100' --fs $fs \
    -o ~/Desktop/figs/lc100_hv_bars.pdf

# EA vs NEA
python scripts/data_display/hypervolume.py \
    result_csvs/ppo/bco_pareto_* result_csvs/ppo/neural_bco_pareto_* --fs $fs \
    --nc 'LC-100' --nc 'EA' --nc 'NEA' -o ~/Desktop/figs/nea_hv_bars.pdf

# Laval
python scripts/data_display/plot_pareto.py \
    result_csvs/dec7results/stl_laval_1.0.csv \
    result_csvs/ppo/s100_laval_* \
    result_csvs/ppo/neural_bco_laval_* --labels \
    --nc 'LC-100' --nc XYZ --nc NEA --nc STL \
    --asymmetric -o ~/Desktop/figs/laval_pareto.pdf
