es=1.5
ns=0
width=8.5
height=10

python scripts/data_display/plot_generated_networks.py cfg/eval/lavalDA.yaml \
    --es $es --ns $ns --width $width --height $height --sc laval_bus_termini.yaml \
    --ls --routes stl_da_routes.pkl -o ../thesis_latex/figs/ch5/stl_routes.pdf
# S100 routes
python scripts/data_display/plot_generated_networks.py cfg/eval/lavalDA.yaml \
    --es $es --ns $ns --width $width --height $height --sc laval_bus_termini.yaml \
    --routes ppo_output_routes/nn_construction_s100_laval_s0_a0.0_routes.pkl \
    --ls -o ../thesis_latex/figs/ch5/lc100_routes_a0.pdf
python scripts/data_display/plot_generated_networks.py cfg/eval/lavalDA.yaml \
    --es $es --ns $ns --width $width --height $height --sc laval_bus_termini.yaml \
    --routes ppo_output_routes/nn_construction_s100_laval_s0_a0.5_routes.pkl \
    --ls -o ../thesis_latex/figs/ch5/lc100_routes_a0p5.pdf
python scripts/data_display/plot_generated_networks.py cfg/eval/lavalDA.yaml \
    --es $es --ns $ns --width $width --height $height --sc laval_bus_termini.yaml \
    --routes ppo_output_routes/nn_construction_s100_laval_s2_a1.0_routes.pkl \
    --ls -o ../thesis_latex/figs/ch5/lc100_routes_a1.pdf
# NEA routes
python scripts/data_display/plot_generated_networks.py cfg/eval/lavalDA.yaml \
    --es $es --ns $ns --width $width --height $height --sc laval_bus_termini.yaml \
    --routes ppo_output_routes/neural_bco_nea_laval_s0_a0.0_routes.pkl \
    --ls -o ../thesis_latex/figs/ch5/nea_routes_a0.pdf
python scripts/data_display/plot_generated_networks.py cfg/eval/lavalDA.yaml \
    --es $es --ns $ns --width $width --height $height --sc laval_bus_termini.yaml \
    --routes ppo_output_routes/neural_bco_nea_laval_s0_a0.5_routes.pkl \
    --ls -o ../thesis_latex/figs/ch5/nea_routes_a0p5.pdf
python scripts/data_display/plot_generated_networks.py cfg/eval/lavalDA.yaml \
    --es $es --ns $ns --width $width --height $height --sc laval_bus_termini.yaml \
    --routes ppo_output_routes/neural_bco_nea_laval_s2_a1.0_routes.pkl \
    --ls -o ../thesis_latex/figs/ch5/nea_routes_a1.pdf
