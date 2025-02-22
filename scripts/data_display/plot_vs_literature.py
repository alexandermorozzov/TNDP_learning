import argparse

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import plotting_utils as pu

# set up matplotlib and seaborn
matplotlib.rcParams['text.usetex'] = True
# force type-1 fonts
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True

matplotlib.rcParams['figure.constrained_layout.use'] = True

# this sets the font in the legend...kind of a kludge, since we don't use 
 # seaborn for anything else
sns.set_theme(style='ticks', palette='colorblind', font='Times New Roman')

HUE_ORDER = ['NEA', 'EA', 'LC-100', 'LC-Greedy', 'Mumford (2013)', 
             'John et al. (2014)', 'K{\\i}l{\\i}\\c{c} and G{\\"o}k 2014', 
             'Ahmed et al. (2019)', 'H{\\"u}sselmann et al. (2023)']
Z_ORDER = ['Mumford (2013)', 'John et al. (2014)', 
           'K{\\i}l{\\i}\\c{c} and G{\\"o}k (2014)', 'Ahmed et al. (2019)', 
           'H{\\"u}sselmann et al. (2023)', 'NEA', 'EA', 'LC-100', 'LC-Greedy']


env_sizes_map = {
    'Mandl': 15,
    'Mumford0': 30,
    'Mumford1': 70,
    'Mumford2': 110,
    'Mumford3': 127,
    'Laval': 632
}


def plot_vs_literature(df, args):
    """For each method in the dataframe, plot the results of the given metric
       (y-axis) against the size of the environment in nodes (x-axis), all on a
       single plot."""
    x_label = '\# nodes $n$'
    df[x_label] = 0
    for key, val in env_sizes_map.items():
        df.loc[df['Environment'] == key, x_label] = val
    
    # keep only the entries in hue order that are in the dataframe
    if args.named_colour:
        # hue_order = [hh if hh in args.named_colour else ''
        #              for hh in HUE_ORDER]
        hue_order = [hh if hh in HUE_ORDER else ''
                     for hh in args.named_colour]
    else:
        hue_order = [hh for hh in HUE_ORDER if any(hh == df['Method'])]

    
    axes = {}
    for method in hue_order:
        method_df = df.loc[df['Method'] == method]
        if method == 'LC-100':
            method = 'LC-100 (ours)'
        elif method == 'NEA':
            method = 'NEA (ours)'
        ax = plt.plot(method_df[x_label], method_df[args.metric], label=method,
                      marker='o', linestyle='--')
        axes[method] = ax[0]
    
    for zorder, method in enumerate(Z_ORDER):
        if method == 'LC-100':
            method = 'LC-100 (ours)'
        elif method == 'NEA':
            method = 'NEA (ours)'
        if method in axes:
            axes[method].zorder = zorder

    plt.xlabel('Number of nodes $n$')
    plt.ylabel(args.metric)

    if not args.nolegend:
        plt.legend()
    plt.yscale('log')
    if args.o:
        plt.savefig(args.o)
        plt.clf()
    else:
        plt.show()
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='path to csv file with data to plot')
    parser.add_argument('metric', help='metric to plot')
    parser.add_argument('-o', help='If provided, save to file')

    # parser.add_argument('--labels', action='store_true',
    #                     help='If provided, print labels on the axes')
    # parser.add_argument('--notitle', action='store_true', 
    #                     help='If provided, do not set a title.')
    # parser.add_argument('--asymmetric', action='store_true', 
    #                     help="If provided, don't halve RTT")
    parser.add_argument('--nolegend', action='store_true', 
                        help="If provided, don't show the legend")
    parser.add_argument('--ms', default=15, type=float, help="Marker size")
    # parser.add_argument('--cs', default=3, type=float,
    #                     help="Error bar cap size")
    parser.add_argument('--fs', default=3, type=float, help="Font size")
    parser.add_argument('--sans', action='store_true', 
                        help="If provided, use a sans-serif font")
    parser.add_argument('--named_colour', '--nc', action='append', 
                        help="assign colours to methods in given order")
    # parser.add_argument('--legend_loc', '--ll', default='best', 
    #                     help="location for the legend")
    args = parser.parse_args()



    if args.sans:
        sns.set_theme(style="ticks", palette='colorblind', font='sans-serif')
        matplotlib.rcParams['text.usetex'] = False

    sns.set_context("paper", font_scale=args.fs, 
                    rc={"lines.markersize": args.ms})

    # unified_df = pu.aggregate_and_preprocess_data(args.data, args.asymmetric)

    df = pd.read_csv(args.data, encoding='latin-1')
    # df.rename(columns={'ATT': '$C_p$ (minutes)', 
    #                    'RTT': '$C_o$ (minutes)'}, inplace=True)

    plot_vs_literature(df, args)


if __name__ == "__main__":
    main()
