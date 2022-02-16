import json
import shutil
import argparse
import datetime

from tqdm import tqdm

from tools.supervisely_utils import *

logger = logging.getLogger(__name__)


def utc_now():
    utc_time = datetime.datetime.utcnow()
    utc_time = utc_time.replace(tzinfo=datetime.timezone.utc)
    return utc_time.strftime("%Y-%m-%dT%H:%M:%SZ")


def main(
        input_dir: str,
        classes_border: dict,
        classes_filling: dict,
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

        for idx, row in tqdm(df_dataset.iterrows(), desc='Mask update', unit=' ann'):
            filename = row['filename']
            img_path = row['img_path']
            ann_path = row['ann_path']
            shutil.copyfile(f'{img_path}', f'{output_dir}/{dataset}/img/{os.path.basename(img_path)}')
            f = open(ann_path)
            ann_data = json.load(f)

            if len(ann_data['objects']) > 0:
                new_objects = []
                for obj in ann_data['objects']:
                    if obj['classTitle'] in classes_border:
                        obj_mask64 = obj['bitmap']['data']
                        obj_mask = base64_to_mask(obj_mask64)
                        binary_mask = binary_fill_holes(obj_mask.copy(), structure=None)
                        binary_mask = binary_opening(binary_mask, structure=None)
                        binary_mask = 255 * binary_mask.astype(np.uint8)
                        new_object_mask = binary_mask - obj_mask
                        try:
                            y = np.where(new_object_mask == 255)[0][0]
                            x = np.where(new_object_mask == 255)[1][0]
                            new_object_mask = mask_to_base64(new_object_mask)
                            # # palette = get_palette(class_names)
                            new_objects.append(obj)
                            title = classes_border[obj['classTitle']]
                            new_objects.append(
                                {
                                    # 'id': 1,
                                    'classId': classes_filling[title],
                                    'description': '',
                                    'geometryType': 'bitmap',
                                    'lablerLogin': 'Hunter_911',
                                    'createdAt': utc_now(),
                                    'updatedAt': utc_now(),
                                    'tags': [],
                                    'classTitle': title,
                                    'bitmap': {
                                        'data': new_object_mask,
                                        'origin': [
                                            int(obj['bitmap']['origin'][0]),
                                            int(obj['bitmap']['origin'][1])
                                        ]
                                    }
                                }
                            )
                        except:
                            logger.warning(f'The object {obj["classTitle"]} has no cavity, filename {filename}')
                            continue
                    elif obj['classTitle'] in classes_filling:
                        continue
                    else:
                        new_objects.append(obj)
                ann_data['objects'] = new_objects
                with open(f'{output_dir}/{dataset}/ann/{os.path.basename(ann_path)}', 'w') as outfile:
                    json.dump(ann_data, outfile)
            else:
                shutil.copyfile(f'{ann_path}', f'{output_dir}/{dataset}/ann/{os.path.basename(ann_path)}')


if __name__ == "__main__":

    GROUP_CLASSES_1 = {
        'Capillary wall': 'Capillary lumen',
        'Arteriole wall': 'Arteriole lumen',
        'Venule wall': 'Venule lumen',
    }

    GROUP_CLASSES_2 = {
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
        classes_border=GROUP_CLASSES_1,
        classes_filling=GROUP_CLASSES_2,
        df_project=df,
        output_dir=args.save_dir,
    )