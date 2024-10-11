python scripts/data_display/plot_pareto.py \
    result_csvs/ppo/s100_pareto_1.0.csv \
    result_csvs/march26results/random_constructor_pareto_1.0.csv \
    result_csvs/ppo/s100_pp_pareto_1.0.csv --labels --ll 'right' --fs 2.2 \
    -e Mumford2 -e Mumford3 -o ../thesis_latex/figs/ch2/extremes.pdf
