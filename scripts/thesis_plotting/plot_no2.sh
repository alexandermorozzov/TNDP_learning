python scripts/data_display/plot_pareto.py \
    result_csvs/dec7results/neural_bco_no2_pareto_* \
    result_csvs/dec7results/neural_bco_pareto_* --labels \
    -o ../thesis_latex/figs/ch4/pareto_plots/no2.pdf
python scripts/data_display/hypervolume.py \
    result_csvs/dec7results/neural_bco_no2_pareto_* \
    result_csvs/dec7results/neural_bco_pareto_* \
    -o ../thesis_latex/figs/ch4/no2_hv_bars.pdf
