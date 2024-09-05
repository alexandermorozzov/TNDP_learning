python scripts/data_display/plot_pareto.py \
    result_csvs/dec7results/s100_pareto_* \
    result_csvs/march26results/greedy_pareto_* \
    result_csvs/init_experiments/init_ahmed_pareto_* \
    result_csvs/init_experiments/init_john_pareto_* \
    result_csvs/init_experiments/init_nikolic_pareto_* \
    --nc 'LC-100' --nc 'LC-Greedy' --nc 'Nikolić (2013)' --nc 'Ahmed (2019)' \
    --nc 'John (2014)' --labels -o ../thesis_latex/figs/ch3/init.pdf