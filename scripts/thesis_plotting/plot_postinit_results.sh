for prefix in ea hh; do
    python scripts/data_display/plot_pareto.py \
        result_csvs/init_experiments/init_${prefix}* --labels \
        --nc 'LC-100' --nc 'LC-Greedy' --nc 'Nikolić (2013)' \
        --nc 'Ahmed (2019)' --nc 'John (2014)' \
        -o ../thesis_latex/figs/ch3/${prefix}.pdf
    python scripts/data_display/hypervolume.py \
        result_csvs/init_experiments/init_${prefix}* \
        --nc 'LC-100' --nc 'LC-Greedy' --nc 'Nikolić (2013)' \
        --nc 'Ahmed (2019)' --nc 'John (2014)' \
        -o ../thesis_latex/figs/ch3/${prefix}_hv_bars.pdf
done
