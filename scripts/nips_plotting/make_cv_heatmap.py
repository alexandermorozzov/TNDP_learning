import argparse
from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from aggregate_data import aggregate_and_preprocess_data


# set up matplotlib and seaborn
matplotlib.rcParams['text.usetex'] = True
# force type-1 fonts
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True

matplotlib.rcParams['figure.constrained_layout.use'] = True

sns.set_theme(style='ticks', palette='colorblind', font='Times New Roman')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', nargs='+', 
                        help='path to csv file with data to format')
    parser.add_argument('--fs', default=2, type=float, help="Font size")    
    parser.add_argument('--sans', action='store_true', help="Font size")
    args = parser.parse_args()

    if args.sans:
        sns.set_theme(style="ticks", palette='colorblind', font='sans-serif')
        matplotlib.rcParams['text.usetex'] = False

    sns.set_context("paper", font_scale=args.fs)

    # load the csv files into dataframes
    unified_df = aggregate_and_preprocess_data(args.data)
    cvs = (unified_df['n_uncovered'] > 0) | (unified_df['n_stops_oob'] > 0)
    unified_df['constraints violated'] = cvs.astype(bool)

    # compute the sum of constraints violated for each method in each environment
    grp = unified_df.groupby(['Method', 'Environment', '$\\alpha$']).sum()

    # find the maximum number of constraints violated
    max_cv = grp['constraints violated'].max()
    
    for method in unified_df['Method'].unique():
        # plot a heatmap for this method, with alpha on one axis and 
         # Environment on the other, with the summed number of constraints 
         # violated as the value
        method_df = grp.loc[method]
        method_df = method_df.reset_index()
        method_df = method_df.pivot(index='$\\alpha$', columns='Environment', 
                                    values='constraints violated')
        
        sns.heatmap(method_df, annot=True, fmt='d', vmin=0, vmax=max_cv)
        plt.title(f'{method} constraints violated')
        # rotate the x-axis labels by 45 degrees to make them more readable
        plt.xticks(rotation=45)
        plt.savefig('figs/cv_heatmap_{}.pdf'.format(method))
        # clear the plot for the next method
        plt.clf()

    methods = ['LC-Greedy', 'RC-100']
    fig, ax = plt.subplots(1, len(methods))
    for ii, method in enumerate(methods):
        method_df = grp.loc[method]
        method_df = method_df.reset_index()
        method_df = method_df.pivot(index='$\\alpha$', columns='Environment', 
                                    values='constraints violated')

        # don't show the color bar for all but the last heatmap
        show_cbar = ii == len(methods) - 1
        sns.heatmap(method_df, annot=True, fmt='d', vmin=0, vmax=max_cv, 
                    ax=ax[ii], cbar=show_cbar)
        ax[ii].set_title(f'{method}')
        # only plot y-axis labels and ticks for the first heatmap
        if ii == 0:
            # make the y-axis labels oriented vertically
            ax[ii].yaxis.set_tick_params(rotation=0)
        if ii > 0:
            ax[ii].set_yticks([])
            ax[ii].set_ylabel('')
        
        # don't label the x-axis for any heatmap
        ax[ii].set_xlabel('')
    
    # add a title to the entire figure
    fig.suptitle('\# constraints violated')
    # plt.title('Number of constraints violated by method and environment')
    # plt.subplots_adjust(top=0.85)
    plt.savefig('figs/cv_heatmap_all.pdf')


if __name__ == "__main__":
    main()