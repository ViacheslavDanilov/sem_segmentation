import os
import argparse
import numpy as np
from typing import List

import cv2
import mmcv
from glob import glob

from mmseg.apis import init_segmentor, inference_segmentor


def main(
        args,
        palette: List[List[int]],
) -> None:

    if not os.path.exists(args.save_dir):
        os.makedirs(os.path.join(args.save_dir, 'mask'))
        os.makedirs(os.path.join(args.save_dir, 'union'))

    model = init_segmentor(args.config, args.checkpoint, device='cuda:0') # TODO: доработать момент с выбором девайса

    images_path = glob(args.images_path + '/*.[pj][npe]*')
    images = []
    for idx, img_path in enumerate(images_path):
        img = mmcv.imread(img_path)
        if args.use_patchify_images:
            images = np.hstack(images, 0) # TODO: добавить функцию нарезки по патчам
        else:
            images.append(img)

    results = predict_model(
        model=model,
        images=images,
        batch_size=args.batch_size,
    )

    for res, img, img_name in zip(results, images, images_path):
        if args.use_patchify_images:
            continue # добавить функцию нарезки по патчам
        else:
            img_union = img_mask_union(
                img=img,
                mask=res,
                classes=model.CLASSES,
                palette=model.PALETTE,
            )
            cv2.imwrite(f'{args.save_dir}/union/{os.path.basename(img_name)}', img_union)
            cv2.imwrite(f'{args.save_dir}/mask/{os.path.basename(img_name)}', res)


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
        color_mask[mask == idx] = 255
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
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--images_path', required=True, type=str)
    parser.add_argument('--save_dir', required=True, type=str)
    parser.add_argument('--use_patchify_images', action='store_true')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='id of gpu to use (only applicable to non-distributed testing)')

    main(args=parser.parse_args(),
         palette=PALETTE)
