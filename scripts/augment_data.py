"""Augment data using image manipulation techniques"""

import re
import os
import random
import numpy as np
import imgaug as ia
from tqdm import tqdm
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from config import (
    AUG_DIR,
    data_config
)

import misc

def main():
    """Main Method"""
    # img_for_rotation = 40 # Number of random picks for rotated augmentation
    # rotational_augment_portion = 0.4 # Portion of rotated images of all augmented images
    original_data_percentage = 0.5
    rotation_percentage = 0.5 # Portion of rotated images in augments
    sometimes_prob = 0.5
    random_seed = data_config.get('random_seed', 4321)
    data_dir = data_config['data_dir']
    processed_dir = data_config['processed_dir']
    results_dir = data_config['results_dir']
    data_types = data_config['data_types']
    data_labels = data_config['data_labels']
    im_size = np.array(data_config['processed_dim'])
    rot_size = (im_size + im_size / 2).astype('int')

    ia.seed(random_seed)
    random.seed(random_seed)
    sometimes = lambda aug: iaa.Sometimes(sometimes_prob, aug)

    feature_dir = f'{results_dir}/data_rgb/train'
    target_dir = f'{processed_dir}/train/target/1'

    misc.gen_dirs([f'{AUG_DIR}/{data_labels[0]}', f'{AUG_DIR}/{data_labels[1]}'])

    filenames = sorted(
        [filename for filename in os.listdir(feature_dir) if filename.endswith('jpg')],
        key=lambda x: int(re.search('\d+', x).group())
    )

    raw_data = [
        [fname for fname in os.listdir(data_dir) if fname.endswith(f'{type}.txt')]
        for type in data_types
    ]
    test_size = data_config.get('test_size', 0.2) / len(np.transpose(raw_data))
    train_files, __ = train_test_split(
        np.transpose(filenames),
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
    print("Augmenting images...")

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

    # for idx, image in enumerate(features_aug):
        # plt.imsave(f'{AUG_DIR}/{data_labels[0]}/{idx:05d}_aug.jpg', image)
        # np.save(
            # f'{AUG_DIR}/{data_labels[1]}/{idx:05d}_aug.npy',
            # targets_aug[idx].squeeze()
        # )


    features_rot = np.empty((rot_augments, *rot_size, 3))
    targets_rot = np.empty((rot_augments, *rot_size, 1))

    for idx in range(rot_augments):
        raw_fname = random.choice(train_files)


if __name__ == "__main__":
    misc.to_local_dir(__file__)
    main()
