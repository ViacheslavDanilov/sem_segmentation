import os
import json
import logging
import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from tools.supervisely_utils import read_sly_project

logger = logging.getLogger(__name__)


def main(
        df_project: pd.DataFrame,
        classes: Tuple[str],
        train_size: float,
        test_size: float,
        seed: int,
        save_dir: str,
) -> None:

    val_size = 1 - train_size - test_size
    assert train_size + val_size + test_size == 1, 'The sum of subset ratios must be equal to 1'

    logger.info('Classes              : {}'.format(classes))
    logger.info('Number of classes    : {}'.format(len(classes)))
    logger.info('Output directory     : {}'.format(save_dir))
    logger.info('Train/Val/Test split : {:.2f} / {:.2f} / {:.2f}'.format(train_size, val_size, test_size))
    datasets = list(set(df_project.dataset))

    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()
    for dataset in datasets:
        df_dataset = df_project[df_project.dataset == dataset]
        df_dataset.reset_index(drop=True, inplace=True)
        _df_train, df_val_test = train_test_split(df_dataset, train_size=train_size, random_state=seed)
        _df_val, _df_test = train_test_split(df_val_test, test_size=test_size/(val_size + test_size), random_state=seed)
        logger.info('')
        logger.info('Dataset      : {:s}'.format(dataset))
        logger.info('Total images : {:d}'.format(len(_df_train) + len(_df_val) + len(_df_test)))
        logger.info('Train images : {:d}'.format(len(_df_train)))
        logger.info('Val images   : {:d}'.format(len(_df_val)))
        logger.info('Test images  : {:d}'.format(len(_df_test)))

        df_train = df_train.append(_df_train)
        df_val = df_val.append(_df_val)
        df_test = df_test.append(_df_test)

    df_train.reset_index(drop=True, inplace=True)
    df_train['subset'] = 'train'
    df_val.reset_index(drop=True, inplace=True)
    df_val['subset'] = 'val'
    df_test.reset_index(drop=True, inplace=True)
    df_test['subset'] = 'test'

    logger.info('')
    logger.info('Dataset      : {:s}'.format('Final'))
    logger.info('Total images : {:d}'.format(len(df_train) + len(df_val) + len(df_test)))
    logger.info('Train images : {:d}'.format(len(df_train)))
    logger.info('Val images   : {:d}'.format(len(df_val)))
    logger.info('Test images  : {:d}'.format(len(df_test)))

    df_final = pd.concat([df_train, df_val, df_test], axis=0)
    df_final.reset_index(drop=True, inplace=True)
    save_path = os.path.join(save_dir, 'dataset.xlsx')
    # df_final.to_excel(save_path, sheet_name='Dataset', index=False, startrow=0, startcol=0)   # TODO: Uncomment once the algo is done

    # Create dataset dirs
    subset_dirs = {
        'img_dir': {},
        'ann_dir': {},
    }
    for ds_dir in subset_dirs:
        for subset_dir in ['train', 'val', 'test']:
            _dir = os.path.join(save_dir, ds_dir, subset_dir)
            subset_dirs[ds_dir][subset_dir] = _dir
            os.makedirs(_dir) if not os.path.isdir(_dir) else False

    # TODO: generate and save masks
    for idx, row in tqdm(df_final.iterrows(), desc='Image processing', unit=' images'):
        ann_path = row['ann_path']
        f = open(ann_path)
        ann_data = json.load(f)


if __name__ == '__main__':

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%d.%m.%Y %I:%M:%S',
        filename='logs/{:s}.log'.format(Path(__file__).stem),
        filemode='w',
        level=logging.INFO,
    )

    CLASSES = [
        'Capillary lumen',
        'Capillary wall',
        'Venule lumen',
        'Venule wall',
        'Arteriole lumen',
        'Arteriole wall',
        'Endothelial cell',
        'Pericyte',
        'SMC',
    ]

    parser = argparse.ArgumentParser(description='Dataset conversion')
    parser.add_argument('--prj_dir', required=True, type=str)
    parser.add_argument('--classes', nargs='+', default=CLASSES, type=str)
    parser.add_argument('--include_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--exclude_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--train_size', default=0.80, type=float)
    parser.add_argument('--test_size', default=0.10, type=float)
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--save_dir', required=True, type=str)
    args = parser.parse_args()

    df = read_sly_project(
        project_dir=args.prj_dir,
        include_dirs=args.include_dirs,
        exclude_dirs=args.exclude_dirs,
    )

    main(
        df_project=df,
        classes=tuple(args.classes),
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
        save_dir=args.save_dir,
    )

    logger.info('Dataset conversion complete!')
