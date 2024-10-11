python scripts/data_display/plot_pareto.py \
    result_csvs/dec7results/stl_laval_1.0.csv \
    result_csvs/ppo/s100_laval_* \
    result_csvs/ppo/neural_bco_laval_* \
    result_csvs/ppo/neural_bco_random_laval_1.* --labels \
    --asymmetric -o ../thesis_latex/figs/ch5/laval_pareto.pdf
# python scripts/data_display/hypervolume.py \
#     result_csvs/ppo/s100_laval_* \
#     result_csvs/ppo/neural_bco_laval_* \
#     --asymmetric -o ../thesis_latex/figs/ch5/laval_hv_bars.pdf
