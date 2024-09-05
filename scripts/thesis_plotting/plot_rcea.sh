python scripts/data_display/plot_pareto.py \
    result_csvs/dec7results/neural_bco_pareto_* \
    result_csvs/dec7results/neural_bco_random_pareto_* \
    result_csvs/neural_bco_pptrained_pareto_1.0.csv --labels \
    --nc NEA --nc "$\pi_{\theta_{\alpha = 1}}$ NEA" --nc "RC-EA" \
    -o ../thesis_latex/figs/ch4/pareto_plots/random.pdf
python scripts/data_display/hypervolume.py \
    result_csvs/dec7results/neural_bco_pareto_* \
    result_csvs/dec7results/neural_bco_random_pareto_* \
    --nc NEA --nc "$\pi_{\theta_{\alpha = 1}}$ NEA" --nc "RC-EA" \
    -o ../thesis_latex/figs/ch4/random_hv_bars.pdf