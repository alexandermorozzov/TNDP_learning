import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import plotting_utils as pu

# set up matplotlib and seaborn
matplotlib.rcParams['text.usetex'] = True
# force type-1 fonts
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True

matplotlib.rcParams['figure.constrained_layout.use'] = True

sns.set_theme(style='ticks', palette='colorblind', font='Times New Roman')


def pareto_plot_from_dataframe(df, args):
    if args.env is None:
        # If there's more than one, sort them alphabetically...a bit of a 
        #  kludge, since it's only a coincidence that alphabetical order is
        #  the same as number-of-nodes order in this case.
        envs = sorted(df['Environment'].unique())
    else:
        envs = args.env

    if len(envs) == 1:
        ncols = 1
        nrows = 1
    else:
        ncols = 2
        # should be 3 for Mandl + Mumford
        nrows = math.ceil(len(envs) / ncols)

    # print any constraint violations
    has_viol = (df['n_stops_oob'] > 0) | (df['n_uncovered'] > 0)
    if has_viol.any():
        # group by environment and method
        oob_df = df.loc[has_viol]
        grp = oob_df.groupby(['Method', 'Environment', '$\\alpha$'])
        size = grp.size()
        print("Constraint violations:")
        print(size)

    # Filter out any rows that have constraint violations
    df = df.loc[~has_viol]

    # plot all the environments as subplots in one big plot
    subplot_size = matplotlib.rcParams['figure.figsize']
    size = [subplot_size[0] * ncols, subplot_size[1] * nrows]
    fig, axes = plt.subplots(nrows, ncols, figsize=size)
    # plt.subplots_adjust(wspace=0.5, hspace=0.5)
    if len(envs) == 1:
        flat_axes = [axes]
    else:
        flat_axes = axes.flatten()
    for ax, env in zip(flat_axes, envs):
  
        env_df = df.loc[df['Environment'] == env]
        if len(env_df) == 0:
            continue

        style = 'Method'

        grp = env_df.groupby(['Method', 'Environment', '$\\alpha$'])
        mean = grp.mean()

        method_set = set(mean.index.get_level_values(0))
        hue_order = [hh for hh in pu.HUE_ORDER if hh in method_set]
        unknown_methods = method_set - set(hue_order)
        hue_order += list(unknown_methods)

        # plot error bars
        std = grp.std()
        # if there is only one row in the group, std will be NaN, so make it 0
        std = std.fillna(0)
        zipped = zip(mean.groupby('Method'), std.groupby('Method'))
        data_by_method = {mm: (md, sd) for (mm, md), (_, sd) in zipped}

        palette = pu.get_dict_palette(args)

        for method in hue_order:
            # iterate over hue order so the dots and errorbars have the same
                # colours
            if method not in data_by_method:
                continue

            md, sd = data_by_method[method]
            if not sd.isnull().all().all():
                if isinstance(palette, dict):
                    color = palette[method]
                else:
                    color = None
                ax.errorbar(md['$C_p$ (minutes)'], md['$C_o$ (minutes)'],
                            xerr=sd['$C_p$ (minutes)'],
                            yerr=sd['$C_o$ (minutes)'], 
                            fmt='o', capsize=args.cs, capthick=1, markersize=0,
                            ecolor=color)
            
        if args.named_colour:
            # markers = list(range(len(palette)))
            markers = True
            style_order = [nc if nc in data_by_method else '' 
                           for nc in args.named_colour]
        else:
            markers = True
            style_order = hue_order
        # to get legend handles and labels, we have to set legend=True...
        g1 = sns.lineplot(x='$C_p$ (minutes)', y='$C_o$ (minutes)', 
                          hue='Method', hue_order=hue_order, data=mean, 
                          markers=markers, style=style, style_order=style_order,
                          markersize=args.ms, legend=True, palette=palette, 
                          ax=ax)
        # ...but we then delete it because we want to put the legend elsewhere
        g1.get_legend().remove()
        
        for paper_name in pu.PAPER_NAME_MAP.values():
            if paper_name in data_by_method:
                ax.scatter(mean.loc[paper_name]['$C_p$ (minutes)'],
                           mean.loc[paper_name]['$C_o$ (minutes)'],
                           marker='o', label=paper_name)

        # use range axes a la Edward Tufte
        ymax = (mean['$C_o$ (minutes)'] + std['$C_o$ (minutes)']).max()
        ymin = (mean['$C_o$ (minutes)'] - std['$C_o$ (minutes)']).min()

        xmax = (mean['$C_p$ (minutes)'] + std['$C_p$ (minutes)']).max()
        xmin = (mean['$C_p$ (minutes)'] - std['$C_p$ (minutes)']).min()
                
        pu.set_tufte_spines(ax, xmin, xmax, ymin, ymax)

        # don't put axis labels on the subplots
        g1.set(xlabel=None, ylabel=None)

        if not args.notitle:
            ax.set_title(env)

    # don't show default axes for the empty subplots
    n_extra_axes = ncols * nrows - len(envs)
    if n_extra_axes > 0:
        for ax in flat_axes[-n_extra_axes:]:
            ax.axis('off')

    if args.labels:
        # put axis labels on the whole figure
        fig.supxlabel('$C_p$ (minutes)')
        fig.supylabel('$C_o$ (minutes)')

    # get the handles and labels from the last subplot
    if not args.nolegend:
        if len(envs) > len(flat_axes):
            legend_loc = 'center'
        else:
            legend_loc = args.legend_loc

        handles, labels = g1.get_legend_handles_labels()
        # put the legend in the last subplot
        flat_axes[-1].legend(handles, labels, frameon=False, loc=legend_loc)

    if args.o:
        plt.savefig(args.o)
        plt.clf()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', nargs='+', 
                        help='path to csv file with data to plot')
    parser.add_argument('-o', help='If provided, save to file')
    parser.add_argument('--labels', action='store_true',
                        help='If provided, print labels on the axes')
    parser.add_argument('--notitle', action='store_true', 
                        help='If provided, do not set a title.')
    parser.add_argument('--asymmetric', action='store_true', 
                        help="If provided, don't halve RTT")
    parser.add_argument('--nolegend', action='store_true', 
                        help="If provided, don't show the legend")
    parser.add_argument('--ms', default=15, type=float, help="Marker size")
    parser.add_argument('--cs', default=3, type=float,
                        help="Error bar cap size")
    parser.add_argument('--fs', default=3, type=float, help="Font size")
    parser.add_argument('--sans', action='store_true', 
                        help="If provided, use a sans-serif font")
    parser.add_argument('--named_colour', '--nc', action='append', 
                        help="assign colours to methods in given order")
    parser.add_argument('-e', '--env', action='append',
                        help='If provided, plot only the provided environments')
    parser.add_argument('--legend_loc', '--ll', default='best', 
                        help="location for the legend")
    args = parser.parse_args()

    if args.sans:
        sns.set_theme(style="ticks", palette='colorblind', font='sans-serif')
        matplotlib.rcParams['text.usetex'] = False

    sns.set_context("paper", font_scale=args.fs, 
                    rc={"lines.markersize": args.ms})

    unified_df = pu.aggregate_and_preprocess_data(args.data, args.asymmetric)

    unified_df.rename(columns={'ATT': '$C_p$ (minutes)', 
                               'RTT': '$C_o$ (minutes)'}, inplace=True)

    pareto_plot_from_dataframe(unified_df, args)


if __name__ == "__main__":
    main()
