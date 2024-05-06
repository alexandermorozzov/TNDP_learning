import argparse

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from aggregate_data import aggregate_and_preprocess_data, PAPER_NAME_MAP

# set up matplotlib and seaborn
matplotlib.rcParams['text.usetex'] = True
# force type-1 fonts
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True

matplotlib.rcParams['figure.constrained_layout.use'] = True

sns.set_theme(style='ticks', palette='colorblind', font='Times New Roman')


HUE_ORDER = ['NEA', 'EA', 'LC-100', 'LC-Greedy', 'LC-40k', 'all-1 NEA', 
             'PC-EA', 'NREA', 'TF', 'STL', 'RC-100', 'RCa-100']
# HUE_ORDER += list(PAPER_NAME_MAP.values())


def pareto_plot_from_dataframe(df, args, env, out_path=None):
    all_env_dfs = []
    env_df = df.loc[df['Environment'] == env]
    all_env_dfs.append(env_df)
    df = pd.concat(all_env_dfs)

    style = 'Method'

    grp = df.groupby(['Method', 'Environment', '$\\alpha$'])
    mean = grp.mean()

    method_set = set(mean.index.get_level_values(0))
    hue_order = [hh for hh in HUE_ORDER if hh in method_set]
    unknown_methods = method_set - set(hue_order)
    hue_order += list(unknown_methods)

    # uncomment the below to get the alternative style
    # styles = ['solid', 'dashed', 'dotted', 'dashdot']
    # plot error bars
    std = grp.std()
    zipped = zip(mean.groupby('Method'), std.groupby('Method'))
    data_by_env = {ee: (md, sd) for (ee, md), (_, sd) in zipped}
    for method in hue_order:
        # iterate over hue order so the dots and errorbars have the same
            # colours
        if method not in data_by_env:
            continue
        md, sd = data_by_env[method]
        if not sd.isnull().all().all():
            plt.errorbar(md['$C_p$ (minutes)'], md['$C_o$ (minutes)'],
                         sd['$C_o$ (minutes)'], sd['$C_p$ (minutes)'],
                         fmt='o', capsize=args.cs, capthick=1, markersize=0,)

    g1 = sns.lineplot(x='$C_p$ (minutes)', y='$C_o$ (minutes)', 
                      hue='Method', style=style, hue_order=hue_order,
                      style_order=hue_order, markers=True, data=mean, 
                      markersize=args.ms, legend=not args.nolegend)
    
    for paper_name in PAPER_NAME_MAP.values():
        if paper_name in data_by_env:
            plt.scatter(mean.loc[paper_name]['$C_p$ (minutes)'],
                        mean.loc[paper_name]['$C_o$ (minutes)'],
                        marker='o', label=paper_name)

    # get the current axis of matplotlib
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if not args.labels:
        g1.set(xlabel=None, ylabel=None)

    if not args.nolegend:
        plt.legend(frameon=False)

    if args.title:
        plt.title(args.title)

    if out_path:
        plt.savefig(out_path)
        plt.clf()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', nargs='+', 
                        help='path to csv file with data to plot')
    parser.add_argument('-e', '--env', action='append',
                        help='If provided, plot only this environment')
    parser.add_argument('-o', help='If provided, save to file')
    parser.add_argument('--labels', action='store_true',
                        help='If provided, print labels on the axes')
    parser.add_argument('--title', help='If provided, set title on figure')
    parser.add_argument('--asymmetric', action='store_true', 
                        help="If provided, don't halve RTT")
    parser.add_argument('--nolegend', action='store_true', 
                        help="If provided, don't show the legend")
    parser.add_argument('--ms', default=15, type=float, help="Marker size")
    parser.add_argument('--cs', default=3, type=float,
                        help="Error bar cap size")
    parser.add_argument('--fs', default=2, type=float,
                        help="Font size")
    parser.add_argument('--sans', action='store_true', help="Font size")
    args = parser.parse_args()

    if args.sans:
        sns.set_theme(style="ticks", palette='colorblind', font='sans-serif')
        matplotlib.rcParams['text.usetex'] = False

    sns.set_context("paper", font_scale=args.fs, 
                    rc={"lines.markersize": args.ms})

    unified_df = aggregate_and_preprocess_data(args.data, args.asymmetric)

    unified_df.rename(columns={'ATT': '$C_p$ (minutes)', 
                               'RTT': '$C_o$ (minutes)'}, inplace=True)

    if args.env is None:
        envs = unified_df['Environment'].unique()
    else:
        # it's just a single environment
        envs = args.env

    # plot each environment separately
    for env in envs:
        if args.o:
            if len(envs) > 1:
                # alter the name for the different environments
                body, _, ext = args.o.rpartition('.')
                out_path = body + f'_{env}.{ext}'
            else:
                out_path = args.o
        else:
            out_path = None

        pareto_plot_from_dataframe(unified_df, args, env, out_path)


if __name__ == "__main__":
    main()