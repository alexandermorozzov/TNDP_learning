python scripts/data_display/plot_pareto.py \
    result_csvs/dec7results/stl_laval_1.0.csv \
    result_csvs/dec7results/s100_laval_* \
    result_csvs/dec7results/neural_bco_laval_* \
    result_csvs/dec7results/neural_bco_random_laval_* --labels \
    --asymmetric -o ../thesis_latex/figs/ch5/laval_pareto.pdf
python scripts/data_display/hypervolume.py \
    result_csvs/dec7results/stl_laval_1.0.csv \
    result_csvs/dec7results/s100_laval_* \
    result_csvs/dec7results/neural_bco_laval_* \
    result_csvs/dec7results/neural_bco_random_laval_* \
    --asymmetric -o ../thesis_latex/figs/ch5/laval_hv_bars.pdf
