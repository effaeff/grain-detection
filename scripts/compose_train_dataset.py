"""Script to compose train dataset so that only a portion of the original data size is used"""

import os
import re
import shutil
import matplotlib.pyplot as plt

import argparse

from tqdm import tqdm

from config import PROCESSED_DIR

def main():
    """Main method"""
    source_dir = '/cephfs/grain_dataset/02_processed/train'

    parser = argparse.ArgumentParser()
    parser.add_argument("data_percentage", type=float)
    args = parser.parse_args()
    data_percentage = args.data_percentage

    print(f"Composing train dataset using {data_percentage*100}% of original data size")

    # Remove all train data first
    for fname in os.listdir(f'{PROCESSED_DIR}/train/features/1'):
        os.remove(f'{PROCESSED_DIR}/train/features/1/{fname}')
    for fname in os.listdir(f'{PROCESSED_DIR}/train/target/1'):
        os.remove(f'{PROCESSED_DIR}/train/target/1/{fname}')

    train_names = sorted(
        os.listdir(f'{source_dir}/features/1'), key=lambda x: int(re.search('\d+', x).group())
    )
    n_used = len(train_names) * data_percentage
    n_used = int(2 * round(n_used / 2))

    print(f"Number of used training examples: {n_used}")
    for fname in tqdm(train_names[:n_used]):
        shutil.copyfile(f'{source_dir}/features/1/{fname}', f'{PROCESSED_DIR}/train/features/1/{fname}')
        shutil.copyfile(f'{source_dir}/target/1/{fname}', f'{PROCESSED_DIR}/train/target/1/{fname}')

if __name__ == "__main__":
    main()
