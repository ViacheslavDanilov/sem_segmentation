import os
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Coverter logs json to csv')
    parser.add_argument('--json-path', default='../work_dirs/fcn_m-v2-d8_512x1024_80k_cityscapes/None.log.json')
    # parser.add_argument('--json-path', default='../work_dirs/20220210_011806.log.json')
    parser.add_argument('--output-dir', default='../work_dirs/fcn_m-v2-d8_512x1024_80k_cityscapes')
    parser.add_argument('--colums-train', default=['memory',
                                                   'data_time',
                                                   'decode.loss_ce',
                                                   'decode.acc_seg',
                                                   'aux.loss_ce',
                                                   'aux.acc_seg',
                                                   'loss'
                                                   ])
    parser.add_argument('--colums-obligatory', default=['mode',
                                                        'epoch',
                                                        'iter',
                                                        'lr'])
    parser.add_argument('--colums-delete', default=['seed',
                                                    'env_info',
                                                    'exp_name',
                                                    'mmseg_version',
                                                    'config',
                                                    'CLASSES',
                                                    'PALETTE',
                                                    'loss',
                                                    'time'])
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    dataframe = pd.read_json(args.json_path, lines=True)
    for mode in dataframe['mode'].unique():
        if mode == mode:
            df = dataframe.loc[dataframe['mode'] == mode]
            if mode != 'train':
                df = df.drop(columns=[col for col in list(dataframe.columns) if col in args.colums_train +
                                      args.colums_delete], axis=1)
            else:
                df = df.drop(columns=[col for col in list(dataframe.columns) if col not in args.colums_train +
                                      args.colums_obligatory], axis=1)
            df.to_csv(f'{args.output_dir}/{mode}.csv')