import json
from typing import Any, Dict

import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tqdm import tqdm

from src.data.utils_sly import *

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def log_dataset(
    dataset_name: str,
    train_images: int,
    val_images: int,
    test_images: int,
) -> None:

    log.info('')
    log.info(f'Dataset...............: {dataset_name}')
    log.info(f'Total images..........: {train_images + val_images + test_images}')
    log.info(f'Train images..........: {train_images}')
    log.info(f'Val images............: {val_images}')
    log.info(f'Test images...........: {test_images}')


def split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.80,
    test_size: float = 0.10,
    seed: int = 11,
):

    val_size = 1 - train_size - test_size
    assert train_size + val_size + test_size == 1, 'The sum of subset ratios must be equal to 1'

    datasets = list(set(df.dataset))
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()
    for dataset in datasets:
        df_dataset = df[df.dataset == dataset]
        df_dataset.reset_index(drop=True, inplace=True)
        _df_train, df_val_test = train_test_split(
            df_dataset,
            train_size=train_size,
            random_state=seed,
        )
        _df_val, _df_test = train_test_split(
            df_val_test,
            test_size=test_size / (val_size + test_size),
            random_state=seed,
        )
        log_dataset(dataset, len(_df_train), len(_df_val), len(_df_test))
        df_train = pd.concat([df_train, _df_train])
        df_val = pd.concat([df_val, _df_val])
        df_test = pd.concat([df_test, _df_test])

    df_train.sort_values(by=['img_path'], inplace=True)
    df_train.reset_index(drop=True, inplace=True)
    df_train['subset'] = 'train'

    df_val.sort_values(by=['img_path'], inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    df_val['subset'] = 'val'

    df_test.sort_values(by=['img_path'], inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    df_test['subset'] = 'test'

    return df_train, df_val, df_test


def save_metadata(
    df: pd.DataFrame,
    save_dir: str,
) -> None:

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'metadata.xlsx')
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    df.to_excel(
        save_path,
        sheet_name='Data',
        index=True,
    )


def create_subset_dirs(
    save_dir: str,
) -> None:

    subdirs: Dict[str, Dict[Any, Any]] = {
        'img_dir': {},
        'ann_dir': {},
    }
    for ds_dir in subdirs:
        for subdir in ['train', 'val', 'test']:
            _dir = os.path.join(save_dir, ds_dir, subdir)
            subdirs[ds_dir][subdir] = _dir
            os.makedirs(_dir, exist_ok=True)


def process_dataset(
    df: pd.DataFrame,
    class_names: Tuple[str],
    palette: List[List[int]],
    save_dir: str,
    exclude_empty_masks: bool = False,
    use_smoothing: bool = False,
) -> np.array:

    data = np.array([], dtype=np.uint8)
    for idx, row in tqdm(df.iterrows(), desc='Image processing', unit=' images'):
        filename = row['filename']
        img_path = row['img_path']
        ann_path = row['ann_path']
        f = open(ann_path)
        ann_data = json.load(f)
        img_size = (
            ann_data['size']['height'],
            ann_data['size']['width'],
        )

        # Iterate over objects
        mask = np.zeros(img_size, dtype=np.uint8)

        for obj in ann_data['objects']:
            class_name = obj['classTitle']
            if class_name in class_names:
                class_id = class_names.index(class_name)
                obj_mask64 = obj['bitmap']['data']
                obj_mask = base64_to_mask(obj_mask64)
                if use_smoothing:
                    obj_mask = smooth_mask(obj_mask)
                obj_mask = obj_mask.astype(float)
                obj_mask *= class_id / np.max(obj_mask)
                obj_mask = obj_mask.astype(np.uint8)

                mask = insert_mask(
                    mask=mask,
                    obj_mask=obj_mask,
                    origin=obj['bitmap']['origin'],
                )
        log.debug('Empty mask: {:s}'.format(Path(filename).name))

        if np.sum(mask) == 0 and exclude_empty_masks:
            pass
        else:
            subset = row['subset']

            # Save image
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img_save_path = os.path.join(save_dir, 'img_dir', subset, f'{filename}.png')
            cv2.imwrite(img_save_path, img)

            # Accumulate masks for class weight calculation
            mask_vector = np.concatenate(mask)
            data = np.append(data, mask_vector, axis=0)

            # Save colored mask
            mask = Image.fromarray(mask).convert('P')
            mask.putpalette(np.array(palette, dtype=np.uint8))
            mask_save_path = os.path.join(save_dir, 'ann_dir', subset, f'{filename}.png')
            mask.save(mask_save_path)
    return data


def compute_class_weights(
    data: np.ndarray,
    class_names: List[str],
    palette: List[List[int]],
    save_dir: str,
):

    num_classes = np.unique(data)
    try:
        class_weights = class_weight.compute_class_weight(
            y=data,
            classes=num_classes,
            class_weight='balanced',
        )
        class_weights = list(class_weights)
        class_weights = [round(x, 2) for x in class_weights]
        log.info('')
        log.info(f'Class weights: {class_weights}')
    except Exception as e:
        class_weights = np.full([len(num_classes)], np.nan)
        log.info('')
        log.warning(f'Class weights: {class_weights}')

    class_meta = {}
    for name, color, weight in zip(class_names, palette, class_weights):
        _class_meta = {
            'color': color,
            'weight': weight,
        }
        class_meta[name] = _class_meta

    class_meta_path = os.path.join(save_dir, 'class_meta.json')
    with open(class_meta_path, 'w') as f:
        json.dump(class_meta, f)


@hydra.main(config_path=os.path.join(os.getcwd(), 'config'), config_name='data', version_base=None)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    df_sly = read_sly_project(
        project_dir=cfg.convert.project_dir,
        include_dirs=cfg.convert.include_dirs,
        exclude_dirs=cfg.convert.exclude_dirs,
    )

    val_size = 1 - cfg.convert.train_size - cfg.convert.test_size

    log.info(f'Classes...............: {cfg.convert.class_names}')
    log.info(f'Number of classes.....: {len(cfg.convert.class_names)}')
    log.info(f'Exclude empty masks...: {cfg.convert.exclude_empty_masks}')
    log.info(f'Output directory......: {cfg.convert.save_dir}')
    log.info(
        f'Train/Val/Test split..: {cfg.convert.train_size:.2f} / {val_size:.2f} / {cfg.convert.test_size:.2f}',
    )

    # Split dataset and save its metadata
    df_train, df_val, df_test = split_dataset(
        df=df_sly,
        train_size=cfg.convert.train_size,
        test_size=cfg.convert.test_size,
        seed=cfg.convert.seed,
    )
    df_final = pd.concat([df_train, df_val, df_test], axis=0)

    # Save metadata
    save_metadata(
        df=df_final,
        save_dir=cfg.convert.save_dir,
    )
    log_dataset('Final', len(df_train), len(df_val), len(df_test))

    # Create dataset structure
    create_subset_dirs(cfg.convert.save_dir)

    # Process dataset iterating over image-annotation pairs
    palette = get_palette(cfg.convert.class_names)
    data = process_dataset(
        df=df_final,
        class_names=cfg.convert.class_names,
        palette=palette,
        exclude_empty_masks=cfg.convert.exclude_empty_masks,
        use_smoothing=cfg.convert.use_smoothing,
        save_dir=cfg.convert.save_dir,
    )

    # Compute class weights
    compute_class_weights(
        data=data,
        class_names=cfg.convert.class_names,
        palette=palette,
        save_dir=cfg.convert.save_dir,
    )

    log.info('')
    log.info('Complete')


if __name__ == '__main__':
    main()
