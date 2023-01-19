import base64
import io
import logging
import os
import zlib
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import supervisely_lib as sly
from PIL import Image
from scipy.ndimage import binary_fill_holes, binary_opening


def get_class_color(
    class_name: str,
) -> List[int]:

    try:
        mapping_dict = {
            'Background': [128, 128, 128],
            'Capillary lumen': [105, 45, 33],
            'Capillary wall': [196, 156, 148],
            'Venule lumen': [31, 119, 180],
            'Venule wall': [174, 199, 232],
            'Arteriole lumen': [212, 0, 2],
            'Arteriole wall': [255, 124, 121],
            'Endothelial cell': [227, 119, 194],
            'Pericyte': [150, 240, 52],
            'SMC': [144, 19, 254],
        }
        return mapping_dict[class_name]
    except Exception as e:
        raise ValueError(f'Unrecognized class_name: {class_name}')


def get_palette(
    class_names: Tuple[str],
) -> List[List[int]]:

    palette = []
    for class_name in class_names:
        class_color = get_class_color(class_name)
        palette.append(class_color)

    return palette


def read_sly_project(
    project_dir: str,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> pd.DataFrame:

    logging.info(f'Processing of {project_dir}')
    assert os.path.exists(project_dir) and os.path.isdir(
        project_dir,
    ), f'Wrong project dir: {project_dir}'
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
                f'Skip {Path(dataset_name).name} because it is not in the include_datasets list',
            )
            continue
        if exclude_dirs and dataset_name in exclude_dirs:
            logging.info(
                f'Skip {Path(dataset_name).name} because it is in the exclude_datasets list',
            )
            continue

        for item_name in dataset:
            img_path, ann_path = dataset.get_item_paths(item_name)
            filename = Path(img_path).stem
            mask_name = f'{filename}.png'
            mask_path = os.path.join(dataset.directory, 'masks_machine', mask_name)

            filenames.append(filename)
            img_paths.append(img_path)
            mask_paths.append(mask_path)
            ann_paths.append(ann_path)
            dataset_names.append(dataset_name)

    df = pd.DataFrame.from_dict(
        {
            'img_path': img_paths,
            'ann_path': ann_paths,
            'mask_path': mask_paths,
            'dataset': dataset_names,
            'filename': filenames,
        },
    )

    return df


def mask_to_base64(mask: np.array):
    img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
    img_pil.putpalette([0, 0, 0, 255, 255, 255])
    bytes_io = io.BytesIO()
    img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
    bytes = bytes_io.getvalue()
    return base64.b64encode(zlib.compress(bytes)).decode('utf-8')


def base64_to_mask(s: str) -> np.ndarray:
    z = zlib.decompress(base64.b64decode(s))
    n = np.frombuffer(z, np.uint8)
    img_decoded = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
    if (len(img_decoded.shape) == 3) and (img_decoded.shape[2] >= 4):
        mask = img_decoded[:, :, 3].astype(np.uint8)  # 4-channel images
    elif len(img_decoded.shape) == 2:
        mask = img_decoded.astype(np.uint8)  # 1-channel images
    else:
        raise RuntimeError('Wrong internal mask format')
    return mask


def smooth_mask(
    binary_mask: np.ndarray,
    fill_holes: bool = False,
) -> np.ndarray:
    if fill_holes:
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5, 5))
        binary_mask = binary_fill_holes(binary_mask, structure=kernel)
    binary_mask = binary_opening(binary_mask, structure=None)
    binary_mask = 255 * binary_mask.astype(np.uint8)
    return binary_mask


def insert_mask(
    mask: np.ndarray,
    obj_mask: np.ndarray,
    origin: List[int],
) -> np.ndarray:

    x, y = origin
    obj_mask_height = obj_mask.shape[0]
    obj_mask_width = obj_mask.shape[1]

    for idx_y in range(obj_mask_height):
        for idx_x in range(obj_mask_width):
            pixel_value = obj_mask[idx_y, idx_x]
            # Check if it is a zero-intensity pixel
            if np.sum(pixel_value) != 0:
                mask[idx_y + y, idx_x + x] = pixel_value

    return mask
