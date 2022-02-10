import os
import argparse

import pandas as pd

if __name__ == "__main__":
    COLUMS_TRAIN = ['memory',
                    'data_time',
                    'decode.loss_ce',
                    'decode.acc_seg',
                    'aux.loss_ce',
                    'aux.acc_seg',
                    'loss']
    COLUMS_OBLIGATORY = ['mode',
                         'epoch',
                         'iter',
                         'lr']
    COLUMS_DELETE = ['seed',
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
    parser.add_argument('--colums_train', default=COLUMS_TRAIN)
    parser.add_argument('--colums_obligatory', default=COLUMS_OBLIGATORY)
    parser.add_argument('--colums_delete', default=COLUMS_DELETE)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataframe = pd.read_json(args.json_path, lines=True)
    for mode in ['train', 'val']:
        df = dataframe.loc[dataframe['mode'] == mode]
        if mode != 'train':
            df = df.drop(columns=[col for col in list(dataframe.columns) if col in args.colums_train +
                                  args.colums_delete], axis=1)
        else:
            df = df.drop(columns=[col for col in list(dataframe.columns) if col not in args.colums_train +
                                  args.colums_obligatory], axis=1)
        df.to_csv(f'{args.save_dir}/{mode}.csv')
