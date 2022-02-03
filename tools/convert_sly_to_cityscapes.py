import json
import logging
import argparse
from typing import List
from pathlib import Path

import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from tools.supervisely_utils import read_project_as_dataframe

logger = logging.getLogger(__name__)


def main(
        project_df: pd.DataFrame,
        train_size: float,
        test_size: float,
        seed: int,
        save_dir: str,
) -> None:

    val_size = 1 - train_size - test_size
    assert train_size + val_size + test_size == 1, 'The sum of subset ratios must be equal to 1'

    logger.info('Input directory(s)   : {}'.format(project_df))
    logger.info('Output directory     : {}'.format(save_dir))
    logger.info('Train/Val/Test split : {:.2f} / {:.2f} / {:.2f}'.format(train_size, val_size, test_size))

    # TODO: split dataset


if __name__ == '__main__':

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%d.%m.%Y %I:%M:%S',
        filename='logs/{:s}.log'.format(Path(__file__).stem),
        filemode='w',
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description='Dataset conversion')
    parser.add_argument('--prj_dir', required=True, type=str)
    parser.add_argument('--include_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--exclude_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--train_size', default=0.80, type=float)
    parser.add_argument('--test_size', default=0.10, type=float)
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--save_dir', required=True, type=str)
    args = parser.parse_args()

    df = read_project_as_dataframe(
        project_dir=args.prj_dir,
        include_dirs=args.include_dirs,
        exclude_dirs=args.exclude_dirs,
    )

    main(
        project_df=df,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
        save_dir=args.save_dir,
    )

    logger.info('Dataset conversion complete!')
