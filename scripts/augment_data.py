"""Augment data using image manipulation techniques"""

import re
import os
import random
import numpy as np
import imgaug as ia
from glob import glob
from tqdm import tqdm
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from pytorchutils.globals import torch, DEVICE
from pytorchutils.ahg import AHGModel

from config import (
    AUG_DIR,
    data_config,
    model_config
)

import misc

def main():
    """Main Method"""
    original_data_percentage = 0.5
    rotation_percentage = 1 # Portion of rotated images in augments
    sometimes_prob = 0.5
    random_seed = data_config.get('random_seed', 4321)
    data_dir = data_config['data_dir']
    processed_dir = data_config['processed_dir']
    model_dir = model_config['models_dir']
    results_dir = data_config['results_dir']
    data_types = data_config['data_types']
    data_labels = data_config['data_labels']
    im_size = np.array(data_config['processed_dim'])
    rot_size = (im_size + im_size / 2).astype('int')

    checkpoint = glob(f'{model_dir}/best/*Model*')[0]
    model = AHGModel(model_config).to(DEVICE)
    state = torch.load(checkpoint)
    model.load_state_dict(state['state_dict'])

    ia.seed(random_seed)
    random.seed(random_seed)
    sometimes = lambda aug: iaa.Sometimes(sometimes_prob, aug)

    feature_dir = f'{results_dir}/data_rgb/train'
    target_dir = f'{processed_dir}/train/target/1'

    misc.gen_dirs([f'{AUG_DIR}/{data_labels[0]}', f'{AUG_DIR}/{data_labels[1]}'])

    filenames = sorted(
        [filename for filename in os.listdir(target_dir) if filename.endswith('.npy')],
        key=lambda x: int(re.search('\d+', x).group())
    )

    raw_fnames = [
        sorted([
            fname for fname in os.listdir(data_dir) if fname.endswith(f'{type}.txt')
        ], key=lambda x: int(re.search('\d+', x).group()))
        for type in data_types
    ]
    test_size = data_config.get('test_size', 0.2) / len(np.transpose(raw_fnames))
    train_files, __ = train_test_split(
        np.transpose(raw_fnames),
        test_size=test_size,
        random_state=random_seed
    )

    seq = iaa.Sequential(
        [
            iaa.SomeOf((4, 8), [
                iaa.HorizontalFlip(1.0),
                iaa.VerticalFlip(1.0),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # gaussian noise
                iaa.Add((-20, 20), per_channel=0.5), # change brightness (by -10 to 10 of original value)
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.OneOf([
                    iaa.GaussianBlur((0, 2.0)), # blur with a sigma between 0 and 2.0
                    iaa.AverageBlur(k=(2, 7)), # blur using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur using local medians with kernel sizes between 3 and 11
                ]),
                iaa.OneOf([
                    iaa.ElasticTransformation(alpha=20, sigma=2),
                    iaa.ElasticTransformation(alpha=30, sigma=3),
                    iaa.ElasticTransformation(alpha=40, sigma=4),
                    iaa.ElasticTransformation(alpha=50, sigma=5)
                ]),
                iaa.AddToHueAndSaturation((-5, 5), per_channel=True),
                iaa.LogContrast(gain=(0.7, 1.1), per_channel=True),
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
                iaa.Crop((0, 50))
            ], random_order=True)  # apply augmenters in random order
        ],
        random_order=True
    )

    total_augments = int(len(filenames) // original_data_percentage - len(filenames))
    rot_augments = int(total_augments * rotation_percentage)
    normal_augments = int(total_augments - rot_augments)

    aug_factor = 1 // original_data_percentage
    normal_aug_factor = normal_augments / len(filenames)

    print(
        f"Original images: {len(filenames)},\t"
        f"Augmentation factor: {aug_factor},\t"
        f"Total augments: {total_augments},\t"
        f"Rotated augments: {rot_augments},\t"
        f"Non-rotated augments: {normal_augments}"
    )

    # rand_fnames = [random.choice(filenames) for __ in range(normal_augments)]

    # features = np.array([
        # plt.imread(f'{feature_dir}/{filename}') for filename in rand_fnames
    # ])
    # targets = np.array([
        # np.load(f'{target_dir}/{os.path.splitext(filename)[0]}.npy')
        # for filename in rand_fnames
    # ])
    # targets = np.reshape(targets.astype('int32'), (*targets.shape, 1))

    # features_aug, targets_aug = seq(images=features, segmentation_maps=targets)


    print("Load raw data...")
    raw_data = [
        np.moveaxis([
            np.loadtxt(
                f'{data_dir}/{filename}',
                delimiter=data_config['delimiter']
            )
            for filename in train_file
        ], 0, -1) # (H, W, C)
        for train_file in tqdm(train_files)
    ]

    print("Augmenting images...")

    features_aug, targets_aug = None, None
    if normal_aug_factor > 0:
        features, targets = get_rand_rgb(raw_data, model, normal_augments, im_size)
        features_aug, targets_aug = seq(images=features, segmentation_maps=targets)

    features_rot_aug, targets_rot_aug = None, None
    if rot_augments > 0:
        features_rot, targets_rot = get_rand_rgb(raw_data, model, rot_augments, rot_size)
        seq.insert(0, iaa.Affine(rotate=(-90, 90)))
        features_rot_aug, targets_rot_aug = seq(images=features_rot, segmentation_maps=targets_rot)
        features_rot_aug = features_rot_aug[
            :,
            rot_size[0]//2 - im_size[0]//2:rot_size[0]//2 + im_size[0]//2,
            rot_size[1]//2 - im_size[1]//2:rot_size[1]//2 + im_size[1]//2,
            :
        ]
        targets_rot_aug = targets_rot_aug[
            :,
            rot_size[0]//2 - im_size[0]//2:rot_size[0]//2 + im_size[0]//2,
            rot_size[1]//2 - im_size[1]//2:rot_size[1]//2 + im_size[1]//2
        ]

    features_comb = (
        features_aug if features_aug is not None else
        features_rot_aug if features_rot_aug is not None else
        np.concatenate((features_aug, features_rot_aug))
    )
    targets_comb = (
        targets_aug if targets_aug is not None else
        targets_rot_aug if targets_rot_aug is not None else
        np.concatenate((targets_aug, targets_rot_aug))
    )

    for idx, image in enumerate(features_comb):
        plt.imsave(f'{AUG_DIR}/{data_labels[0]}/{idx:05d}_aug.jpg', image)
        np.save(
            f'{AUG_DIR}/{data_labels[1]}/{idx:05d}_aug.npy',
            targets_comb[idx].squeeze()
        )

def get_rand_rgb(raw_data, model, n_augments, im_size, use_model=False):
    """Get random rgb image using model evaluation with corresponding mask"""
    features = np.empty((n_augments, *im_size, 3), dtype=np.uint8)
    targets = np.empty((n_augments, *im_size, 1), dtype=np.int32)
    print(f"Sampling {n_augments} images")
    for idx in tqdm(range(n_augments)):
        rand_raw = random.choice(raw_data)
        sample_height = random.randint(0, rand_raw.shape[0] - im_size[0])
        sample_width = random.randint(0, rand_raw.shape[1] - im_size[1])
        rand_sample = rand_raw[
            sample_height:sample_height + im_size[0],
            sample_width:sample_width + im_size[1]
        ]

        rand_features = rand_sample[:, :, :model_config['n_channels']]

        features_rgb = np.concatenate((rand_features, np.zeros((*im_size, 1))), axis=2)
        if use_model:
            rand_features = torch.from_numpy(np.moveaxis(rand_features, -1, 0)).float().to(DEVICE)
            features_rgb = model.inp2rgb(rand_features).detach().cpu().numpy()
            features_rgb = np.moveaxis(features_rgb, 0, -1)
        features_rgb = (features_rgb * 255).astype('uint8')

        rand_targets = rand_sample[:, :, model_config['n_channels']]
        rand_targets = np.reshape(rand_targets, (*rand_targets.shape, 1)).astype('int32')

        features[idx] = features_rgb
        targets[idx] = rand_targets

    return features, targets

if __name__ == "__main__":
    misc.to_local_dir(__file__)
    main()
