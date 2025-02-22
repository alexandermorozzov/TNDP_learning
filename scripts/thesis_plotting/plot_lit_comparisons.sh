# LC-100 C_p
python scripts/data_display/plot_vs_literature.py \
    lc100_results_table.csv '$C_p$ (minutes)' --fs 2 \
    -o ~/Desktop/figs/lc100_results/vs_lit_passenger.pdf
# LC-100 C_o
python scripts/data_display/plot_vs_literature.py \
    lc100_results_table.csv '$C_o$ (minutes)' --fs 2 \
    -o ~/Desktop/figs/lc100_results/vs_lit_operator.pdf --nolegend
# NEA C_p
python scripts/data_display/plot_vs_literature.py nea_results_table.csv \
    '$C_p$ (minutes)' --fs 2 -o ~/Desktop/figs/nea_results/vs_lit_passenger.pdf
# NEA C_o
python scripts/data_display/plot_vs_literature.py nea_results_table.csv \
    '$C_o$ (minutes)' --fs 2 -o ~/Desktop/figs/nea_results/vs_lit_operator.pdf \
    --nolegend
