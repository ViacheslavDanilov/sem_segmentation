import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

import cv2
import numpy as np
from glob import glob

try:
    import pytesseract
except ModuleNotFoundError:
    pass


def get_file_list(
    src_dirs: Union[List[str], str],
    ext_list: Union[List[str], str],
    dirname_template: str = '',
    filename_template: str = '',
) -> List[str]:
    """
    Args:
        src_dirs: directory(s) with files inside
        ext_list: extension(s) used for a search
        dirname_template: include directories with this template
        filename_template: include files with this template
    Returns:
        all_files: a list of file paths
    """
    all_files = []
    src_dirs = [src_dirs, ] if isinstance(src_dirs, str) else src_dirs
    ext_list = [ext_list, ] if isinstance(ext_list, str) else ext_list
    for src_dir in src_dirs:
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                file_ext = Path(file).suffix
                file_ext = file_ext.lower()
                dir_name = os.path.basename(root)
                if (
                        file_ext in ext_list
                        and dirname_template in dir_name
                        and filename_template in file

                ):
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)
    all_files.sort()
    return all_files


def get_dir_list(
    data_dir: str,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> List[str]:
    dir_list = []
    _dir_list = glob(data_dir + '/*/')
    for series_dir in _dir_list:
        if include_dirs and Path(series_dir).name not in include_dirs:
            logging.info(
                'Skip {:s} because it is not in the included_dirs list'.format(
                    Path(series_dir).name
                )
            )
            continue

        if exclude_dirs and Path(series_dir).name in exclude_dirs:
            logging.info(
                'Skip {:s} because it is in the excluded_dirs list'.format(
                    Path(series_dir).name
                )
            )
            continue

        dir_list.append(series_dir)
    return dir_list


def extract_modality_info(
    img: np.ndarray,
    x_lims: Tuple[int] = (0.0, 0.5),
    y_lims: Tuple[int] = (0.9375, 1.0),
) -> Dict:

    keys = [
        'energy',
        'magnification',
    ]
    info = {key: float('nan') for key in keys}

    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        y1 = int(y_lims[0] * img.shape[0])
        y2 = int(y_lims[1] * img.shape[0])
        x1 = int(x_lims[0] * img.shape[1])
        x2 = int(x_lims[1] * img.shape[1])
        img_info = img[y1:y2, x1:x2]
        img_info_inv = 255 - img_info
        txt_info = pytesseract.image_to_string(img_info_inv)
        txt_info = txt_info.strip().split()
        info['energy'] = txt_info[1]
        info['magnification'] = txt_info[2]

    except Exception as e:
        logging.debug('Failed to retrieve modality info')

    return info
