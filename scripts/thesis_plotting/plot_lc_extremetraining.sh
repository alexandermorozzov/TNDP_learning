python scripts/data_display/plot_pareto.py \
    result_csvs/dec7results/s100_pareto_1.0.csv \
    result_csvs/march26results/random_constructor_pareto_1.0.csv \
    result_csvs/extreme_csvs/s100_pp_pareto_1.0.csv --labels \
    -e Mumford2 -o ../thesis_latex/figs/ch2/extremes_Mumford2.pdf
python scripts/data_display/plot_pareto.py \
    result_csvs/dec7results/s100_pareto_1.0.csv \
    result_csvs/march26results/random_constructor_pareto_1.0.csv \
    result_csvs/extreme_csvs/s100_pp_pareto_1.0.csv --nolegend --labels \
    -e Mumford3 -o ../thesis_latex/figs/ch2/extremes_Mumford3.pdf 
