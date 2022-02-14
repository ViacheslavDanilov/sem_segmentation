from typing import List


import mmcv
import argparse
from glob import glob
from mmcv.utils import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmseg.models import build_segmentor
from mmseg.apis import inference_segmentor, show_result_pyplot


def main(
        args,
        palette: List[List[int]],
):

    cfg = Config.fromfile(args.config)
    model = init_model(
        args=args,
        cfg=cfg,
        palette=palette,
    )

    predict_model(
        model=model,
        images_path=glob(args.images_path + '/*.[pj][npe]*'),
        batch_size=args.batch_size,
    )


def init_model(
        args,
        cfg,
        palette: List[List[int]]):
    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]

    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cfg = cfg

    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        model.PALETTE = palette
    return model


def predict_model(
        model,
        images_path: List[str],
        batch_size,
):
    predicts = []
    # TODO: Настроить batch обработку
    # images_batches = []
    # images_batch = []
    # for idx, img_path in enumerate(images_path):
    #     images_batch.append(mmcv.imread(img_path))
    #     if idx % batch_size == 0:
    #         images_batches.append(images_batch)
    #         images_batch = []
    # if len(images_batch) > 0:
    #     images_batches.append(images_batch)
    #
    # for img_batch in images_batches:
    #     result = inference_segmentor(model, img_batch)
    #     predicts += result
    for img_path in images_path:
        img = mmcv.imread(img_path)
        res = inference_segmentor(model, img)
        predicts.append(res)
    # plt.figure(figsize=(8, 6))
    # show_result_pyplot(model, img, result, PALETTE)

    # return mask, union


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
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='id of gpu to use (only applicable to non-distributed testing)')

    main(args=parser.parse_args(),
         palette=PALETTE)
