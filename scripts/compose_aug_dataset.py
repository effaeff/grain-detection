"""Script to compose a dataset including augments using a specific augmentation factor"""

import os
import re
import shutil
import numpy as np
import matplotlib.pyplot as plt

import argparse

from tqdm import tqdm
from glob import glob

from config import PROCESSED_DIR, AUG_DIR

def main():
    """Main method"""

    parser = argparse.ArgumentParser()
    parser.add_argument("data_percentage", type=float)
    args = parser.parse_args()
    original_data_percentage = args.data_percentage

    print(f"Composing train dataset using {original_data_percentage*100}% original data")

    old_aug_f = glob(f'{PROCESSED_DIR}/train/features/1/*_aug.npy')
    old_aug_t = glob(f'{PROCESSED_DIR}/train/target/1/*_aug.npy')
    old_vqgan_f = glob(f'{PROCESSED_DIR}/train/features/1/*_vqgan.npy')
    old_vqgan_t = glob(f'{PROCESSED_DIR}/train/target/1/*_vqgan.npy')

    for f_list in [old_aug_f, old_aug_t, old_vqgan_f, old_vqgan_t]:
        for fname in f_list:
            os.remove(fname)

    n_train = len(os.listdir(f'{PROCESSED_DIR}/train/features/1'))
    augments = int((n_train // original_data_percentage - n_train) // 2)
    augments = int(2 * round(augments / 2))

    print(f'Number of augments: {augments*2}')

    aug_names = sorted(
        os.listdir(f'{AUG_DIR}/features_aug'), key=lambda x: int(re.search('\d+', x).group())
    )
    vqgan_names = sorted(
        os.listdir(f'{AUG_DIR}/features_vqgan'), key=lambda x: int(re.search('\d+', x).group())
    )

    print("Copy augmentations based on image manupulations...")
    for fname in tqdm(aug_names[:augments]):
        shutil.copyfile(f'{AUG_DIR}/features_aug/{fname}', f'{PROCESSED_DIR}/train/features/1/{fname}')
        shutil.copyfile(f'{AUG_DIR}/target_aug/{fname}', f'{PROCESSED_DIR}/train/target/1/{fname}')
    print("Copy VQ-GAN samples...")
    for fname in tqdm(vqgan_names[:augments]):
        shutil.copyfile(f'{AUG_DIR}/features_vqgan/{fname}', f'{PROCESSED_DIR}/train/features/1/{fname}')
        shutil.copyfile(f'{AUG_DIR}/target_vqgan/{fname}', f'{PROCESSED_DIR}/train/target/1/{fname}')

if __name__ == "__main__":
    main()
