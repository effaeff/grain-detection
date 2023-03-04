"""Method to augment images through image processing approaches"""

import random
import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa

from matplotlib import pyplot as plt


def augment_images(data, orig_size, random_seed=1234):
    """
    Augment images using imgaug module

    Args:
        - data: has shape (N, 2, H, W)
          and is assumed to contain image and segmentation map for each sample
        - orig_size: portion of original data in combined dataset
    """
    ia.seed(random_seed)
    print("Augmenting images...")
    n_augments = round(len(data) / orig_size - len(data))
    n_augments_per_image = round(n_augments / len(data))
    print(f"# augments: {n_augments}")

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        iaa.SomeOf((3, 7), [
            iaa.HorizontalFlip(1.0),
            iaa.VerticalFlip(1.0),
            # Add gaussian noise to images
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            # Change brightness of images (by -10 to 10 of original value)
            iaa.Add((-20, 20), per_channel=0.5),
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
            iaa.OneOf([
                iaa.GaussianBlur((0, 2.0)), # Blur images with a sigma between 0 and 3.0
                # Blur image using local means with kernel sizes between 2 and 7
                iaa.AverageBlur(k=(2, 7)),
                # Blur image using local medians with kernel sizes between 2 and 7
                iaa.MedianBlur(k=(3, 11)),
            ]),
            iaa.AddToHueAndSaturation((-5, 5), per_channel=True),
            iaa.LogContrast(gain=(0.7, 1.1), per_channel=True),
            # Sometimes move parts of the image around
            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
            iaa.Crop((0, 50))
        ], random_order=True) # Apply augmenters in random order
    ], random_order=True)

    augments = np.empty((n_augments, data.shape[1], data.shape[2], data.shape[3], 3))
    for idx, sample in enumerate(data):
        if len(sample.shape) > 2: # Segmentation map is present for each image
            print("test")
            image = sample[0, :, :, :3]
            image_map = sample[1, :, :, :3]
            images_base = np.asarray([image for __ in range(n_augments_per_image)])
            maps_base = np.array([image_map for __ in range(n_augments_per_image)])

            aug_im, aug_map = seq(images=images_base, segmentation_maps=maps_base)

            __, axs = plt.subplots(2, 1)
            axs[0].imshow(sample[0])
            axs[1].imshow(sample[1])
            plt.show()

            idx_start = idx * n_augments_per_image
            augments[idx_start:idx_start + n_augments_per_image, 0] = np.array(aug_im)
            augments[idx_start:idx_start + n_augments_per_image, 1] = np.array(aug_map)

    return augments
