"""Script to compose a dataset including augments using a specific augmentation factor"""

import os
import re
import shutil
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from config import PROCESSED_DIR, AUG_DIR

def main():
    """Main method"""
    original_data_percentage = 0.1

    n_train = len(os.listdir(f'{PROCESSED_DIR}/train/features/1'))
    augments = int(n_train // original_data_percentage - n_train)

    aug_names = sorted(
        os.listdir(f'{AUG_DIR}/features'), key=lambda x: int(re.search('\d+', x).group())
    )

    for fname in tqdm(aug_names[:augments]):
        shutil.copyfile(f'{AUG_DIR}/features/{fname}', f'{PROCESSED_DIR}/train/features/1/{fname}')
        shutil.copyfile(f'{AUG_DIR}/target/{fname}', f'{PROCESSED_DIR}/train/target/1/{fname}')

if __name__ == "__main__":
    main()
