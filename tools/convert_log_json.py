import os
import argparse

import pandas as pd


if __name__ == "__main__":

    COLUMNS = [
        'mode',
        'epoch',
        'iter',
        'lr',
        'memory',
        'data_time',
        'time',
        'decode.loss_ce',
        'decode.acc_seg',
        'aux.loss_ce',
        'aux.acc_seg',
        'loss',
    ]

    parser = argparse.ArgumentParser(description='Convert JSON logs to CSV')
    parser.add_argument('--json_path', required=True, type=str)
    parser.add_argument('--save_dir', required=True, type=str)
    parser.add_argument('--columns_train', default=COLUMNS)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataframe = pd.read_json(args.json_path, lines=True)
    for mode in dataframe['mode'].unique():
        if mode == mode:
            df = dataframe.loc[dataframe['mode'] == mode]
            if mode != 'train':
                df = df.drop(columns=args.columns_train+['mode'], axis=1)
            else:
                df = df.drop(columns=[col for col in list(dataframe.columns) if col not in args.columns_train], axis=1)
            df.to_csv(f'{args.save_dir}/{mode}.csv')
