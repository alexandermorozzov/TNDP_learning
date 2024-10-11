import os
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import argparse
from tensorboard.backend.event_processing import event_accumulator

from plotting_utils import set_tufte_spines


# set up seaborn
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['figure.constrained_layout.use'] = True

sns.set_theme(style="ticks", palette='colorblind', font='serif')


def extract_stat_from_event_file(event_file_path, stat, statname):
    """
    Extracts a specific statistic from a TensorBoard event file.

    :param event_file_path: Path to the event file.
    :param stat: The name of the statistic to extract.
    :return: A pandas DataFrame with two columns: step and value for the 
        statistic.
    """
    # Load the TensorBoard event file
    ea = event_accumulator.EventAccumulator(event_file_path)
    ea.Reload()

    # Extract the scalar
    if stat in ea.scalars.Keys():
        scalar_events = ea.Scalars(stat)
        data = {
            "Statistic": statname,
            "step": [s.step for s in scalar_events],
            "value": [s.value for s in scalar_events],
        }
        return pd.DataFrame(data)
    else:
        print(f"Statistic '{stat}' not found in {event_file_path}.")
        return pd.DataFrame(columns=["step", "value"])

def plot_stat(directory, stats, statnames=None, plot_types=None, 
              output_path=None, ylabel=None):
    """
    Plots the mean curve of a specific statistic from TensorBoard logs, with 
    shading indicating one standard deviation above and below the mean.

    :param directory: The directory containing sub-directories of TensorBoard 
        logs.
    :param stat: The statistic to plot.
    :param statname: The name of the statistic to display on the plot.
    :param ylabel: The label for the y-axis.
    """
    if not statnames:
        statnames = stats

    assert len(stats) <= 2, "Can only plot up to two statistics at a time."

    # Plotting
    plt.figure(figsize=(10, 6))
    if plot_types is None:
        plot_types = "l" * len(stats)

    axes = [plt.gca(), plt.twinx()]
    # get the collection of colours from the seaborn palette
    colours = sns.color_palette("colorblind", len(stats))

    for stat, statname, plot_type, ax, ax_side, colour in \
        zip(stats, statnames, plot_types, axes, ['left', 'right'], colours):
        all_data = []
        ax.set_ylabel(statname)

        # Walk through all sub-directories to find event files
        for root, _, files in os.walk(directory):
            for file in files:
                if file.startswith("events.out.tfevents"):
                    path = os.path.join(root, file)
                    df = extract_stat_from_event_file(path, stat, statname)
                    if not df.empty:
                        all_data.append(df)

        if not all_data:
            print("No data found.")
            continue

        # Combine all dataframes and calculate mean and std dev
        print(f"Found {len(all_data)} event files with stat \"{stat}\".")
        all_data = pd.concat(all_data)
        if plot_type == 'l':
            gg = sns.lineplot(data=all_data, x="step", y="value", 
                              estimator="mean", errorbar="sd", label=statname,
                              ax=ax, color=colour)
        elif plot_type == 's':
            mean = all_data.groupby('step').mean()
            sd = all_data.groupby('step').std()
            eb = plt.errorbar(mean.index, mean['value'], yerr=sd['value'],
                              fmt='o', capsize=4, capthick=2, elinewidth=2,
                              markersize=7, label=statname, ax=ax,
                              color=colour)
            plt.plot(mean.index, mean['value'], linestyle='dotted', 
                     linewidth=1.5, color=eb.lines[0].get_color(), ax=ax)

        leg = ax.legend()
        leg.remove()
        ax.yaxis.label.set_color(colour)
        xs = all_data['step']
        # ys = all_data['value']
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_bounds(xs.min(), xs.max())
        # print(ys.min(), ys.max())
        # print(ax_side)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)

    # get the current axis of matplotlib
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    xs = all_data['step']
    ax.spines['bottom'].set_bounds(xs.min(), xs.max())
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_bounds(ys.min(), ys.max())
    # ax.spines['right'].set_bounds(ys.min(), ys.max())

    plt.xlabel("# of training episodes")

    if output_path is None:
        # plt.title(f"Mean and Standard Deviation of {statname} Over Time")
        plt.show()
    else:
        plt.savefig(output_path)
        plt.clf()


def main():
    # read in a directory and a stat from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='path to directory with event files')
    parser.add_argument('-s', '--stat', action='append', 
                        help='name of the statistic to plot')
    parser.add_argument('--statname', '--sn', action='append', 
                        help='if provided, an alternate name to display on '\
                        'the plot for the statistic')
    parser.add_argument('-y', help='Y-axis label for the plot')
    parser.add_argument('--ms', default=15, type=float, help="Marker size")
    parser.add_argument('--fs', default=2, type=float,
                        help="Font size")
    parser.add_argument('-t', 
                        help="Type-of-plot string.  Accepts 'l' for line " \
                            "plot, or 's' for scatter, in a sequence with " \
                            "one character for every --stat arguement.")
    # add an argument for a path to which to save the file
    parser.add_argument('-o', help='If provided, save to file')
    args = parser.parse_args()

    fonts = [ff for ff in matplotlib.font_manager.findSystemFonts()]

    sns.set_context("paper", font_scale=args.fs, 
                    rc={"lines.markersize": args.ms})

    # run the command with the given arguments
    # TODO allow multiple arguments for stats
    plot_types = args.t
    plot_stat(args.directory, args.stat, args.statname, plot_types, 
              output_path=args.o, ylabel=args.y)


if __name__ == "__main__":
    main()