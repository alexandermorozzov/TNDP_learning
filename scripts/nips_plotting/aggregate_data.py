from pathlib import Path
import pandas as pd


HEADER_NAMES = ['$\\alpha$', '# routes', 'cost', 'ATT', 'RTT', '$d_0$', 
                '$d_1$', '$d_2$', '$d_{un}$', 'n_uncovered', 'n_stops_oob', 
                'Time', '# iterations']


# PAPER_NAME_MAP = {
#     'mumford': 'Mumford (2013)',
#     'john': 'John et al. (2014)',
#     'kilic': 'Kılıç and Gök (2014)',
#     'lin': 'Lin and Tang (2022)'
# }

PAPER_NAME_MAP = {
    'mumford': '[63]',
    'john': '[64]',
    'kilic': '[65]',
    'lin': '[66]'
}


def assign_method(df, filename):
    # add a column to each dataframe with the method name
    if filename.startswith('bco'):
        df['Method'] = 'EA'
    elif 'neural_bco_no2' in filename:
        df['Method'] = 'all-1 NEA'
    elif filename.startswith('neural_bco_random'):
        df['Method'] = 'PC-EA'
    elif filename.startswith('neural_bco_short'):
        df['Method'] = 'NEA (short)'
    elif filename.startswith('neural_bco') or filename.startswith('nbco'):
        df['Method'] = 'NEA'
    elif filename.startswith('s40k'):
        df['Method'] = 'LC-40k'
    elif filename.startswith('s100_pp'):
        df['Method'] = 'LC-100, $\\alpha = 1$ policy'
    elif filename.startswith('s100_op'):
        df['Method'] = 'LC-100 $\\alpha = 0$ policy'
    elif filename.startswith('s100'):
        df['Method'] = 'LC-100'
    elif filename.startswith('greedy_pp'):
        df['Method'] = 'LC-Greedy $\\alpha = 1$ policy'
    elif filename.startswith('greedy_op'):
        df['Method'] = 'LC-Greedy $\\alpha = 0$ policy'
    elif filename.startswith('greedy'):
        df['Method'] = 'LC-Greedy'
    elif filename.startswith('tf'):
        df['Method'] = 'TF'
    elif filename.startswith('nrea'):
        df['Method'] = 'NREA'
    elif filename.startswith('stl'):
        df['Method'] = 'STL'
    elif filename.startswith('random_constructor_alphahalt'):
        df['Method'] = 'RCa-100'
    elif filename.startswith('random_init') or \
        filename.startswith('random_constructor'):
        df['Method'] = 'RC-100'
    elif filename.startswith('paper'):
        df['Method'] = PAPER_NAME_MAP[filename.split('_')[1]]
    else:
        df['Method'] = filename.split('_pareto')[0].replace('_', ' ')

    # if not filename.startswith('paper'):
    #     # divide by two because the papers present one-way results
    #     df['RTT'] /= 2


def add_env(df):
    df['Environment'] = 'Mandl'
    df.loc[df['# routes'] == 6, 'Environment'] = 'Mandl'
    df.loc[df['# routes'] == 12, 'Environment'] = 'Mumford0'
    df.loc[df['# routes'] == 15, 'Environment'] = 'Mumford1'
    df.loc[df['# routes'] == 56, 'Environment'] = 'Mumford2'
    df.loc[df['# routes'] == 60, 'Environment'] = 'Mumford3'
    # Laval's STL has 79 asymmetric routes, we have 43 symmetric ones
    df.loc[df['# routes'] == 43, 'Environment'] = 'Laval'
    df.loc[df['# routes'] == 79, 'Environment'] = 'Laval'


def aggregate_and_preprocess_data(data_paths, asymmetric=False):
    dfs = []
    for path in data_paths:
        # load the csv files, making sure they have the correct column names
        if not path.endswith('.csv'):
            continue

        df = pd.read_csv(path, header=0)
        if 'cost' not in df.columns:
            df = pd.read_csv(path, names=HEADER_NAMES)
        if df.columns[0] != '$\\alpha$':
            # set the first column's name to be alpha
            df.rename(columns={df.columns[0]: '$\\alpha$'}, inplace=True)

        filename = Path(path).stem
        last_part = filename.rpartition('_')[-1]
        try:
            df['$\\alpha$'] = float(last_part)
        except ValueError:
            pass
            
        if Path(path).stem.partition('_')[0] == 'neural_bco':
            print(Path(path).stem)

        assign_method(df, filename)
        add_env(df)
        dfs.append(df)

    # concatenate the dataframes
    unified_df = pd.concat(dfs)
    # divide by two because we want to present one-way results
    if not asymmetric:
        unified_df['RTT'] /= 2

    return unified_df


