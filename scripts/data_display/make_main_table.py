import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from plotting_utils import aggregate_and_preprocess_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', nargs='+', 
                        help='path to csv file with data to format')
    parser.add_argument('-v', '--value', default='cost',
                        help='plot this value')
    parser.add_argument('--asymmetric', action='store_true',
                        help="If provided, don't halve RTT")    
    args = parser.parse_args()

    # load the csv files into dataframes
    unified_df = aggregate_and_preprocess_data(args.data, args.asymmetric)

    # select three alpha values to print
    alphas = [0.0, 0.5, 1.0]
    grp_columns = ['Environment', 'Method']
    pivot_index = ['Method']
    for alpha in alphas:
        # select rows in df with this alpha value
        alpha_df = unified_df.loc[unified_df['$\\alpha$'] == alpha]

        grp = alpha_df.groupby(grp_columns)
        # pivot the table so each Environment is a column
        mean = grp.mean().pivot_table(index=pivot_index, 
                                      columns=['Environment'], 
                                      values=args.value)
        std = grp.std().pivot_table(index=pivot_index, columns=['Environment'], 
                                    values=args.value)
        mean = pd.DataFrame([row.apply(lambda xx: '{:.3f}'.format(xx))
                            for _, row in mean.iterrows()])
        std = pd.DataFrame([row.apply(lambda xx: '{:.3f}'.format(xx))
                            for _, row in std.iterrows()])
        both = mean + ' $\pm$ ' + std
        
        print(f'alpha = {alpha}')
        # print the latex table
        print(both.to_latex(float_format='%.3f', escape=False))

        # prepare and print a table of the number of constraints violated
        print('# constraints violated')
        cvs = (alpha_df['n_uncovered'] > 0) | (alpha_df['n_stops_oob'] > 0)
        alpha_df['constraints violated'] = cvs.astype(bool)
        grp = alpha_df.groupby(['Method', 'Environment', '$\\alpha$']).sum()
        pvt = grp.pivot_table(index=pivot_index, columns=['Environment'], 
                              values='constraints violated')
        print(pvt)


if __name__ == "__main__":
    main()
