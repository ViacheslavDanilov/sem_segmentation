import os
import argparse
import numpy as np
from typing import List

import cv2
from glob import glob
from tqdm import tqdm

from mmseg.apis import init_segmentor, inference_segmentor
from supervisely.geometry.sliding_windows import SlidingWindows


def main(
        args,
        palette: List[List[int]],
) -> None:

    patch_size = args.patch_size
    overlap = args.overlap

    if not os.path.exists(args.save_dir):
        os.makedirs(os.path.join(args.save_dir, 'mask'))
        os.makedirs(os.path.join(args.save_dir, 'union'))

    slider = SlidingWindows(
        window_shape=patch_size,
        min_overlap=overlap
    )

    model = init_segmentor(args.config, args.checkpoint, device='cuda:0')
    if model.PALETTE is None:
        model.PALETTE = palette
    images_path = glob(args.images_path + '/*.[pj][npe]*')
    for idx, img_path in tqdm(enumerate(images_path), desc='prediction'):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images_slider = []
        bboxes = []
        for roi_id, roi in enumerate(slider.get(img.shape)):
            x1, x2 = roi.left, roi.left + roi.width
            y1, y2 = roi.top, roi.top + roi.height
            bboxes.append([x1, y1, x2, y2])
            patch = img[y1:y2, x1:x2]
            images_slider.append(patch)

        results_patch = predict_model(
            model=model,
            images=images_slider,
            batch_size=args.batch_size,
            # batch_size=len(images_slider), # 3070 не тянет
        )

        res = np.zeros((img.shape[0], img.shape[1]))
        for box, patch in zip(bboxes, results_patch):
            res[box[1]:box[3], box[0]:box[2]] = patch

        img_union = img_mask_union(
            img=img,
            mask=res,
            classes=model.CLASSES,
            palette=model.PALETTE,
        )
        cv2.imwrite(f'{args.save_dir}/union/{os.path.basename(img_path)}', img_union)
        cv2.imwrite(f'{args.save_dir}/mask/{os.path.basename(img_path)}', res)


def predict_model(
        model,
        images: List[float],
        batch_size: int = 1,
):

    results = inference_segmentor(
        model=model,
        img=images,
        samples_per_gpu=batch_size,
    )
    return results


def img_mask_union(
        img,
        mask,
        classes,
        palette,
):
    for idx, (_, color) in enumerate(zip(classes, palette)):
        color_mask = np.zeros(mask.shape)
        color_mask[mask == idx + 1] = 255
        img = cv2.addWeighted(img, 1, (cv2.cvtColor(color_mask.astype('uint8'), cv2.COLOR_GRAY2RGB)
                                       * color).astype(np.uint8), 0.45, 0)
    return img


if __name__ == "__main__":

    PALETTE = [
        [128, 128, 128],
        [189, 16, 224],
        [139, 87, 42],
        [192, 220, 252],
        [74, 144, 226],
        [250, 177, 186],
        [208, 2, 27]]

    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--patch_size', nargs='+', default=[160, 160], type=int)
    parser.add_argument('--overlap', default=[0, 0], type=int)
    parser.add_argument('--images_path', required=True, type=str)
    parser.add_argument('--save_dir', required=True, type=str)
    parser.add_argument('--use_patchify_images', action='store_true')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='id of gpu to use (only applicable to non-distributed testing)')

    main(args=parser.parse_args(),
         palette=PALETTE)
