import json
import shutil
import argparse

from tqdm import tqdm

from utils import get_current_time
from tools.supervisely_utils import *

logger = logging.getLogger(__name__)


def main(
        input_dir: str,
        class_groups: dict,
        class_ids: dict,
        df_project: pd.DataFrame,
        output_dir: str,
) -> None:
    datasets = list(set(df_project.dataset))
    for dataset in datasets:
        df_dataset = df_project[df_project.dataset == dataset]
        if not os.path.exists(output_dir):
            os.makedirs(os.path.join(output_dir, dataset, 'img'))
            os.makedirs(os.path.join(output_dir, dataset, 'ann'))
        shutil.copyfile(f'{input_dir}/meta.json', f'{output_dir}/meta.json')
        shutil.copyfile(f'{input_dir}/obj_class_to_machine_color.json', f'{output_dir}/obj_class_to_machine_color.json')

        for idx, row in tqdm(df_dataset.iterrows(), desc='Updating Supervisely dataset', unit=' ann'):
            filename = row['filename']
            img_path = row['img_path']
            img_name = Path(img_path).name
            shutil.copyfile(f'{img_path}', f'{output_dir}/{dataset}/img/{img_name}')

            ann_path = row['ann_path']
            ann_name = Path(ann_path).name
            f = open(ann_path)
            ann_data = json.load(f)

            if len(ann_data['objects']) > 0:
                new_objects = []
                for obj in ann_data['objects']:
                    if obj['classTitle'] in class_groups:
                        wall_mask64 = obj['bitmap']['data']
                        wall_mask = base64_to_mask(wall_mask64)
                        filled_mask = binary_fill_holes(wall_mask.copy(), structure=None)
                        filled_mask = 255 * filled_mask.astype(np.uint8)
                        lumen_mask = filled_mask - wall_mask

                        try:
                            # Append a wall object
                            new_objects.append(obj)

                            # Append a lumen object
                            lumen_mask = mask_to_base64(lumen_mask)
                            class_name = class_groups[obj['classTitle']]
                            new_objects.append(
                                {
                                    'classId': class_ids[class_name],
                                    'description': '',
                                    'geometryType': 'bitmap',
                                    'lablerLogin': 'Hunter_911',
                                    'createdAt': get_current_time(),
                                    'updatedAt': get_current_time(),
                                    'tags': [],
                                    'classTitle': class_name,
                                    'bitmap': {
                                        'data': lumen_mask,
                                        'origin': [
                                            int(obj['bitmap']['origin'][0]),
                                            int(obj['bitmap']['origin'][1])
                                        ]
                                    }
                                }
                            )

                        except Exception as e:
                            logger.info(f'The object {obj["classTitle"]} has no cavity. Filename: {filename}')

                    elif obj['classTitle'] in class_ids:
                        continue

                    else:
                        new_objects.append(obj)

                ann_data['objects'] = new_objects
                with open(f'{output_dir}/{dataset}/ann/{ann_name}', 'w') as outfile:
                    json.dump(ann_data, outfile)
            else:
                shutil.copyfile(f'{ann_path}', f'{output_dir}/{dataset}/ann/{ann_name}')


if __name__ == '__main__':

    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%d.%m.%Y %I:%M:%S',
        filename='logs/{:s}.log'.format(Path(__file__).stem),
        filemode='w',
        level=logging.INFO,
    )

    CLASS_GROUPS = {
        'Capillary wall': 'Capillary lumen',
        'Arteriole wall': 'Arteriole lumen',
        'Venule wall': 'Venule lumen',
    }

    CLASS_IDS = {
        'Capillary lumen': 9945624,
        'Arteriole lumen': 9944173,
        'Venule lumen': 9944174,
    }

    parser = argparse.ArgumentParser(description='Update mask dataset')
    parser.add_argument('--project_dir', required=True, type=str)
    parser.add_argument('--include_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--exclude_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--save_dir', required=True, type=str)
    args = parser.parse_args()

    df = read_sly_project(
        project_dir=args.project_dir,
        include_dirs=args.include_dirs,
        exclude_dirs=args.exclude_dirs,
    )

    main(
        input_dir=args.project_dir,
        class_groups=CLASS_GROUPS,
        class_ids=CLASS_IDS,
        df_project=df,
        output_dir=args.save_dir,
    )
