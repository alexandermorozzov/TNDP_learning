python scripts/data_display/plot_pareto.py \
    result_csvs/ppo/s100_pareto_*.csv \
    result_csvs/ppo/bco_pareto_* \
    result_csvs/ppo/neural_bco_pareto_* \
    result_csvs/ppo/s40k_pareto_* --labels \
    -o ../thesis_latex/figs/ch4/pareto_plots/40k.pdf
python scripts/data_display/hypervolume.py \
    result_csvs/ppo/s100_pareto_*csv \
    result_csvs/ppo/bco_pareto_* \
    result_csvs/ppo/neural_bco_pareto_* \
    result_csvs/ppo/s40k_pareto_* \
    --nc NEA --nc EA --nc LC-100 --nc LC-40k \
    -o ../thesis_latex/figs/ch4/40k_hv_bars.pdf
