import json
import argparse

from tqdm import tqdm

from tools.supervisely_utils import *
from tools.utils import extract_modality_info

logger = logging.getLogger(__name__)


def main(
        df_project: pd.DataFrame,
        save_dir: str,
) -> None:

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
        'class_id',
        'obj_height',
        'obj_width',
        'area_abs',
        'area_rel',
    ]

    # Iterate over annotations
    meta = []
    for idx, row in tqdm(df_project.iterrows(), desc='Annotation processing', unit=' JSONs'):
        dataset = row['dataset']
        filename = row['filename']
        img_path = row['img_path']

        ann_path = row['ann_path']
        f = open(ann_path)
        ann_data = json.load(f)
        img_height = ann_data['size']['height']
        img_width = ann_data['size']['width']
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        modality_info = extract_modality_info(img)

        # Iterate over objects
        for obj in ann_data['objects']:
            obj_meta = {key: float('nan') for key in keys}
            class_name = obj['classTitle']
            class_meta = get_class_meta(class_name)
            class_id = class_meta['id']
            obj_mask64 = obj['bitmap']['data']
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
            obj_meta['class_id'] = class_id
            obj_meta['obj_height'] = obj_mask.shape[0]
            obj_meta['obj_width'] = obj_mask.shape[1]
            obj_meta['area_abs'] = area_abs
            obj_meta['area_rel'] = area_rel

            meta.append(obj_meta)

    df_meta = pd.DataFrame(meta)
    df_meta.sort_values(by=['filename'], inplace=True)
    os.makedirs(save_dir) if not os.path.isdir(save_dir) else False
    save_path = os.path.join(save_dir, 'metadata.xlsx')
    df_meta.to_excel(save_path, sheet_name='Metadata', index=False, startrow=0, startcol=0)


if __name__ == '__main__':

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%d.%m.%Y %I:%M:%S',
        filename='logs/{:s}.log'.format(Path(__file__).stem),
        filemode='w',
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description='Metadata extraction')
    parser.add_argument('--project_dir', required=True, type=str)
    parser.add_argument('--include_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--exclude_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--save_dir', required=True, type=str)
    args = parser.parse_args()

    try:
        import pytesseract
    except ModuleNotFoundError:
        logging.info('Extraction of modality info is not available')

    df = read_sly_project(
        project_dir=args.project_dir,
        include_dirs=args.include_dirs,
        exclude_dirs=args.exclude_dirs,
    )

    main(
        df_project=df,
        save_dir=args.save_dir,
    )

    logger.info('Metadata retrieved')
