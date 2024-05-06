import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from aggregate_data import aggregate_and_preprocess_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', nargs='+', 
                        help='path to csv file with data to format')
    parser.add_argument('--asymmetric', action='store_true', 
                        help="If provided, don't halve RTT")
    parser.add_argument('-o', help="prefix of files in which to save output")
    args = parser.parse_args()

    # load the csv files into dataframes
    unified_df = aggregate_and_preprocess_data(args.data, args.asymmetric)
    # halve since we want just one way
    unified_df.rename(columns={'ATT': '$C_p$', 
                               'RTT': '$C_o$'}, inplace=True)
                               
    # select three alpha values to print
    alphas = [0.0, 0.5, 1.0]
    grp_columns = ['Environment', 'Method']
    pivot_index = ['Environment']
    pivot_column = ['Method']
    values = ['$C_p$', '$C_o$', '$d_0$', '$d_1$', '$d_2$', '$d_{un}$', 
            #   'n_uncovered', 'n_stops_oob'
              ]
    for alpha in alphas:
        # select rows in df with this alpha value
        alpha_df = unified_df.loc[unified_df['$\\alpha$'] == alpha]
        alpha_df = alpha_df[['Environment', 'Method'] + values]

        grp = alpha_df.groupby(grp_columns)
        # # pivot the table so each Environment is a column        
        # mean = grp.mean().pivot_table(index=pivot_index, 
        #                               columns=pivot_column, values=values)
        # mean = mean.stack(0).add_suffix(' mean')
        # std = grp.std().pivot_table(index=pivot_index, 
        #                             columns=pivot_column, values=values)
        # std = std.stack(0).add_suffix(' std')
        mean = grp.mean().add_suffix(' mean')
        std = grp.std().add_suffix(' std')
        
        both = pd.concat([mean, std], axis='columns')

        print(f'alpha = {alpha}')

        if args.o:
            out_path = Path(args.o + f'_alpha{alpha}.csv')
            both.to_csv(out_path, float_format='%.2f')
        else:
            print(both)


if __name__ == "__main__":
    main()