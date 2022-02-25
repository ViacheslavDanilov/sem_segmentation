import os
import logging
import argparse
from PIL import Image
from pathlib import Path
from typing import List, Tuple

import cv2
from tqdm import tqdm

from utils import get_file_list
from supervisely.geometry.sliding_windows import SlidingWindows

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %I:%M:%S',
    filename='logs/{:s}.log'.format(Path(__file__).stem),
    filemode='w',
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


def main(
        img_paths: List[str],
        patch_size: Tuple[int],
        overlap: Tuple[int],
        save_dir: str,
):
    logger.info(f'Number of images : {len(img_paths)}')
    logger.info(f'Patch size       : {patch_size}')
    logger.info(f'Patch overlapping: {overlap}')

    os.makedirs(save_dir, exist_ok=True)

    slider = SlidingWindows(
        window_shape=patch_size,
        min_overlap=overlap
    )

    for img_path in tqdm(img_paths, desc='Patchify images', unit=' image'):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img_basename = Path(img_path).stem

        bboxes = []
        for roi_id, roi in enumerate(slider.get(img.shape)):
            x1, x2 = roi.left, roi.left + roi.width
            y1, y2 = roi.top, roi.top + roi.height
            bboxes.append([x1, y1, x2, y2])
            patch = img[y1:y2, x1:x2]
            patch = Image.fromarray(patch).convert('P')
            patch_name = f'{img_basename}_patch_{roi_id}.png'
            save_path = os.path.join(save_dir, patch_name)
            patch.save(save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dataset conversion')
    parser.add_argument('--img_dir', required=True, type=str)
    parser.add_argument('--patch_size', nargs='+', default=[160, 160], type=int)
    parser.add_argument('--overlap', default=[0, 0], type=int)
    parser.add_argument('--save_dir', required=True, type=str)
    args = parser.parse_args()

    img_paths = get_file_list(
        src_dirs=args.img_dir,
        ext_list=[
            '.png',
            '.bmp',
            '.jpg',
            '.jpeg',
        ],
    )

    main(
        img_paths=img_paths,
        patch_size=tuple(args.patch_size),
        overlap=tuple(args.overlap),
        save_dir=args.save_dir,
    )

    logger.info('')
    logger.info('Dataset patchification complete')
