import json

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.utils import extract_modality_info
from src.data.utils_sly import *

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(config_path=os.path.join(os.getcwd(), 'config'), config_name='data', version_base=None)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    try:
        pass
    except ModuleNotFoundError:
        log.warning('Extraction of modality info is not available')

    df_sly = read_sly_project(
        project_dir=cfg.meta.project_dir,
        include_dirs=cfg.meta.include_dirs,
        exclude_dirs=cfg.meta.exclude_dirs,
    )

    keys = [
        'img_path',
        'ann_path',
        'dataset',
        'filename',
        'img_height',
        'img_width',
        'energy',
        'magnification',
        'class',
        'x_c',
        'y_c',
        'obj_height',
        'obj_width',
        'area_abs',
        'area_rel',
    ]

    # Iterate over annotations
    meta = []
    for idx, row in tqdm(df_sly.iterrows(), desc='Annotation processing', unit=' JSONs'):
        dataset = row['dataset']
        filename = row['filename']
        img_path = row['img_path']

        ann_path = row['ann_path']
        f = open(ann_path)
        ann_data = json.load(f)
        img_height = ann_data['size']['height']
        img_width = ann_data['size']['width']
        modality_info = extract_modality_info(img_path)

        # Iterate over objects
        for obj in ann_data['objects']:
            obj_meta = {key: float('nan') for key in keys}
            class_name = obj['classTitle']
            obj_mask64 = obj['bitmap']['data']
            obj_origin = obj['bitmap']['origin']
            obj_mask = base64_to_mask(obj_mask64)
            area_abs = cv2.countNonZero(obj_mask)
            area_rel = area_abs / (img_height * img_width)

            obj_meta['dataset'] = dataset
            obj_meta['img_path'] = img_path
            obj_meta['ann_path'] = ann_path
            obj_meta['img_height'] = img_height
            obj_meta['img_width'] = img_width
            obj_meta['filename'] = filename
            obj_meta['energy'] = modality_info['energy']
            obj_meta['magnification'] = modality_info['magnification']
            obj_meta['class'] = class_name
            obj_meta['x_c'] = obj_origin[0]
            obj_meta['y_c'] = obj_origin[1]
            obj_meta['obj_height'] = obj_mask.shape[0]
            obj_meta['obj_width'] = obj_mask.shape[1]
            obj_meta['area_abs'] = area_abs
            obj_meta['area_rel'] = area_rel

            meta.append(obj_meta)

    df_meta = pd.DataFrame(meta)
    df_meta.sort_values(by=['filename'], inplace=True)
    os.makedirs(cfg.meta.save_dir) if not os.path.isdir(cfg.meta.save_dir) else False
    save_path = os.path.join(cfg.meta.save_dir, 'metadata.xlsx')
    df_meta.to_excel(save_path, sheet_name='Metadata', index=False, startrow=0, startcol=0)
    log.info('Metadata retrieved')


if __name__ == '__main__':
    main()
