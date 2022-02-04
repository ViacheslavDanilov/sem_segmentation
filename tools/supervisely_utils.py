import os
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import supervisely_lib as sly


def get_class_palette(
        class_name: str,
) -> List[int]:

    try:
        mapping_dict = {
            'Capillary lumen': {
                'id': 1,
                'color': [105, 45, 33],
            },
            'Capillary wall': {
                'id': 2,
                'color': [196, 156, 148],
            },
            'Venule lumen': {
                'id': 3,
                'color': [31, 119, 180],
            },
            'Venule wall': {
                'id': 4,
                'color': [174, 199, 232],
            },
            'Arteriole lumen': {
                'id': 5,
                'color': [212, 0, 2],
            },
            'Arteriole wall': {
                'id': 6,
                'color': [255, 124, 121],
            },
            'Endothelial cell': {
                'id': 7,
                'color': [227, 119, 194],
            },
            'Pericyte': {
                'id': 8,
                'color': [150, 240, 52],
            },
            'SMC': {
                'id': 9,
                'color': [144, 19, 254],
            },
        }
        return mapping_dict[class_name]
    except Exception as e:
        raise ValueError('Unrecognized class_name: {:s}'.format(class_name))


def read_sly_project(
    project_dir: str,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None
) -> pd.DataFrame:

    logging.info('Processing of {:s}'.format(project_dir))
    assert os.path.exists(project_dir) and os.path.isdir(project_dir), 'Wrong project dir: {}'.format(project_dir)
    project = sly.Project(
        directory=project_dir,
        mode=sly.OpenMode.READ,
    )

    filenames: List[str] = []
    img_paths: List[str] = []
    mask_paths: List[str] = []
    ann_paths: List[str] = []
    dataset_names: List[str] = []

    for dataset in project:
        dataset_name = dataset.name
        if include_dirs and dataset_name not in include_dirs:
            logging.info(
                'Skip {:s} because it is not in the include_datasets list'.format(
                    Path(dataset_name).name
                )
            )
            continue
        if exclude_dirs and dataset_name in exclude_dirs:
            logging.info(
                'Skip {:s} because it is in the exclude_datasets list'.format(
                    Path(dataset_name).name
                )
            )
            continue

        for item_name in dataset:
            img_path, ann_path = dataset.get_item_paths(item_name)
            filename = Path(img_path).stem
            mask_name = '{:s}.png'.format(filename)
            mask_path = os.path.join(dataset.directory, 'masks_machine', mask_name)

            filenames.append(filename)
            img_paths.append(img_path)
            mask_paths.append(mask_path)
            ann_paths.append(ann_path)
            dataset_names.append(dataset_name)

    df = pd.DataFrame.from_dict({
        'dataset': dataset_names,
        'filename': filenames,
        'img_path': img_paths,
        'mask_path': mask_paths,
        'ann_path': ann_paths,

    })

    return df
