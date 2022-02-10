import os
import argparse

import pandas as pd

if __name__ == "__main__":
    COLUMNS_TRAIN = ['memory',
                    'data_time',
                    'decode.loss_ce',
                    'decode.acc_seg',
                    'aux.loss_ce',
                    'aux.acc_seg',
                    'loss']
    COLUMNS_OBLIGATORY = ['mode',
                         'epoch',
                         'iter',
                         'lr']
    COLUMNS_DELETE = ['seed',
                     'env_info',
                     'exp_name',
                     'mmseg_version',
                     'config',
                     'CLASSES',
                     'PALETTE',
                     'time']

    parser = argparse.ArgumentParser(description='Convert JSON logs to CSV')
    parser.add_argument('--json_path', required=True, type=str)
    parser.add_argument('--save_dir', required=True, type=str)
    parser.add_argument('--columns_train', default=COLUMNS_TRAIN)
    parser.add_argument('--columns_obligatory', default=COLUMNS_OBLIGATORY)
    parser.add_argument('--columns_delete', default=COLUMNS_DELETE)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataframe = pd.read_json(args.json_path, lines=True)
    for mode in ['train', 'val']:
        df = dataframe[dataframe['mode'] == mode]
        if mode != 'train':
            df = df.drop(columns=[col for col in list(dataframe.columns) if col in args.columns_train +
                                  args.columns_delete], axis=1)
        else:
            df = df.drop(columns=[col for col in list(dataframe.columns) if col not in args.columns_train +
                                  args.columns_obligatory], axis=1)
        df.to_csv(f'{args.save_dir}/{mode}.csv')
