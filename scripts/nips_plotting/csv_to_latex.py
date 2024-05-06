import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='path to csv file with data to format')
    parser.add_argument('-i', '--numindex', default=2,
                        help='Number of columns that are index columns')
    parser.add_argument('-s', '--showstd', action='store_true',
                        help='If provided, show std as well as mean')
    args = parser.parse_args()

    # read in the csv file as a pandas dataframe
    index_cols = list(range(args.numindex))
    df = pd.read_csv(args.data, index_col=index_cols, dtype=str)

    mean_cols = sorted([cc for cc in df.columns if cc.endswith(' mean')])
    std_cols = sorted([cc for cc in df.columns if cc.endswith(' std')])

    # for every pair of columns with matching ' mean' and ' std' suffixes:
    for mean_col, std_col in zip(mean_cols, std_cols):
        if args.showstd:
            str_col = df[mean_col] + ' $\pm$ ' + df[std_col]
        else:
            str_col = df[mean_col]
        prefix = mean_col.replace(' mean', '')
        df[prefix] = str_col
        del df[mean_col]
        del df[std_col]
    
    print(df.to_latex(float_format='%.2f', escape=False, na_rep=''))


if __name__ == "__main__":
    main()