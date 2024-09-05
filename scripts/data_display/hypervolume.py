import argparse

from pygmo import hypervolume
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# from nips_plotting.aggregate_data import aggregate_and_preprocess_data, \
#     PAPER_NAME_MAP
# from nips_plotting.plot_pareto import HUE_ORDER, get_dict_palette

import plotting_utils as pu

# set up matplotlib and seaborn
matplotlib.rcParams['text.usetex'] = True
# force type-1 fonts
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True

matplotlib.rcParams['figure.constrained_layout.use'] = True

sns.set_theme(style='ticks', palette='colorblind', font='Times New Roman')

EPSILON = 1e-5

def compute_hypervolume(df, ref_point=None):
    points = df[['ATT', 'RTT']]
    if ref_point is None:
        # add a small margin to the reference point
        ref_point = points.max() + EPSILON
    # remove points that are worse than the reference point
    points = points[(points['ATT'] < ref_point['ATT']) & 
                    (points['RTT'] < ref_point['RTT'])]
    hv = hypervolume(points)
    return hv.compute(ref_point)


def get_pareto_front(df):
    # find the points on the pareto front
    points = []
    atts = df['ATT']
    rtts = df['RTT']

    for idx, row in df.iterrows():
        att = row['ATT']
        rtt = row['RTT']
        worse_somewhere = (att > atts) | (rtt > rtts)
        not_better_anywhere = (rtt >= rtts) & (att >= atts)
        dominated = worse_somewhere & not_better_anywhere
        if not dominated.any():
            points.append((att, rtt))

    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', nargs='+', 
                        help='path to csv file with data to plot')
    parser.add_argument('--asymmetric', action='store_true', 
                        help="If provided, don't halve RTT")
    parser.add_argument('-e', '--env', action='append',
                        help='If provided, plot only this environment')
    parser.add_argument('--named_colour', '--nc', action='append', 
                        help="assign colours to methods in given order")    
    parser.add_argument('--fs', default=2, type=float, help="Font size")
    parser.add_argument('-o', help='If provided, save to file')
    args = parser.parse_args()

    sns.set_context("paper", font_scale=args.fs)

    unified_df = pu.aggregate_and_preprocess_data(args.data, args.asymmetric)
    grp = unified_df.groupby(['Method', 'Environment', '$\\alpha$'])
    mean = grp.mean()

    if args.env is None:
        envs = unified_df['Environment'].unique()
    else:
        envs = args.env

    methods = unified_df['Method'].unique()
    if args.named_colour:
        methods = [mm for mm in args.named_colour if mm in methods]

    hypervolumes = {}
    for env, env_df in mean.groupby('Environment'):
        # find the reference point, which is the maximum of the pareto front

        # first, find the points on the pareto front
        pareto_points = []
        for method in methods:
            method_means = env_df.loc[method]
            # find the points on the pareto front
            pareto_points += get_pareto_front(method_means)

        # then find the reference point from them
        pareto_points = pd.DataFrame(pareto_points, columns=['ATT', 'RTT'])
        ref_point = pareto_points[['ATT', 'RTT']].max() + EPSILON
        print(f'=={env}==')
        env_hvs = {}
        hypervolumes[env] = env_hvs
        for method, env_method_df in env_df.groupby('Method'):
            env_hvs[method] = compute_hypervolume(env_method_df, ref_point)
        print(env_hvs)

    # make a set of bar charts, one for each environment
    palette = pu.get_dict_palette(args)
    ncols = len(envs) + 1 if len(envs) > 1 else 1
    fig_width = 2 * ncols
    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, 6))
    for env, ax in zip(envs, axes[:-1]):
    # for env, env_hvs in hypervolumes.items():
        env_hvs = hypervolumes[env]
        env_hvs_df = pd.DataFrame(env_hvs.items(), 
                                  columns=['Method', 'Hypervolume'])
        hue_order = [hh for hh in pu.HUE_ORDER if hh in methods]
        # add legend and then remove it so the handles and labels are available
        g1 = sns.barplot(env_hvs_df, x='Method', y='Hypervolume', hue='Method',
                         palette=palette, order=hue_order, ax=ax, 
                         legend='full')
        g1.get_legend().remove()
        g1.set(xlabel=None, ylabel=None, xticklabels=[])
        ax.set_title(env)
        ax.tick_params(bottom=False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # put the legend in the last subplot
    if len(envs) > 1:
        legend_loc = 'center'
    else:
        legend_loc = args.legend_loc
    handles, labels = g1.get_legend_handles_labels()
    if args.named_colour:
        # sort the handles and labels in the order of the named colours
        handles = [handles[labels.index(mm)] for mm in methods]
        labels = methods

    axes[-1].axis('off')
    axes[-1].legend(handles, labels, frameon=False, loc=legend_loc)
        
    if args.o:
        plt.savefig(args.o)
        plt.clf()
    else:
        plt.show()

if __name__ == "__main__":
    main()