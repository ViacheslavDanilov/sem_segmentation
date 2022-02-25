import os
import logging
import argparse
from typing import List
from pathlib import Path

import pandas as pd

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %I:%M:%S',
    filename='logs/{:s}.log'.format(Path(__file__).stem),
    filemode='w',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main(
        json_path: str,
        columns_common: List[str],
        columns_train: List[str],
        columns_delete: List[str],
        save_dir: str,
) -> None:

    df_src = pd.read_json(json_path, lines=True)
    for mode in ['train', 'val']:
        df = df_src[df_src['mode'] == mode]
        if mode != 'train':
            df = df.drop(
                columns=[
                    col for col in list(df_src.columns) if col in columns_train + columns_delete
                ],
                axis=1,
            )
        else:
            df = df.drop(
                columns=[
                    col for col in list(df_src.columns) if col not in columns_train + columns_common
                ],
                axis=1,
            )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, '{:s}.xlsx'.format(mode))
        df.to_excel(save_path, sheet_name='Meta', index=False, startrow=0, startcol=0)
        logger.info('{:s} log saved: {:s}'.format(mode.capitalize(), save_path))


if __name__ == "__main__":

    columns_common = [
        'mode',
        'epoch',
        'iter',
        'lr',
    ]

    columns_train = [
        'memory',
        'decode.loss_ce',
        'decode.acc_seg',
        'aux.loss_ce',
        'aux.acc_seg',
        'loss',
    ]

    columns_delete = [
        'seed',
        'env_info',
        'exp_name',
        'data_time',
        'mmseg_version',
        'config',
        'CLASSES',
        'PALETTE',
        'time',
    ]

    parser = argparse.ArgumentParser(description='Convert JSON logs to CSV')
    parser.add_argument('--json_path', required=True, type=str)
    parser.add_argument('--save_dir', required=True, type=str)
    parser.add_argument('--columns_common', default=columns_common)
    parser.add_argument('--columns_train', default=columns_train)
    parser.add_argument('--columns_delete', default=columns_delete)
    args = parser.parse_args()

    main(
        json_path=args.json_path,
        columns_common=args.columns_common,
        columns_train=args.columns_train,
        columns_delete=args.columns_delete,
        save_dir=args.save_dir,
    )
