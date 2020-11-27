"""Data processing methods"""

import re
import os
import glob
import numpy as np
import torchvision.datasets as dset
import matplotlib as mpl
import matplotlib.pyplot as plt

from pytorchutils.globals import nn
from pytorchutils.globals import torch

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torchvision.transforms import Compose, ToTensor, Normalize, Grayscale
from sklearn.model_selection import train_test_split

class DataProcessor(torch.utils.data.Dataset):
    """Class for data processor"""
    def __init__(self, config):
        self.config = config
        self.transform = Compose(
            [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        # Assume binary segmentation as default
        self.output_size = self.config.get('output_size', 2)
        self.batch_size = self.config.get('batch_size', 4)
        self.data_dir = self.config['data_dir']
        self.interim_dir = self.config['interim_dir']
        self.processed_dir = self.config['processed_dir']
        # self.feature_lbl = self.config['feature_label']
        # self.target_lbl = self.config['target_label']
        self.data_lbls = self.config['data_labels']

        filenames = [
            [file for file in os.listdir(self.data_dir) if file.endswith(f'{label}.txt')]
            for label in self.data_lbls
        ]


        for type_idx, files in enumerate(filenames):
            filenames[type_idx].sort(key=lambda f: int(re.sub(r'\D', '', f)))
            Path(
                '{}/{}'.format(self.interim_dir, self.data_lbls[type_idx])
            ).mkdir(parents=True, exist_ok=True)
            if not any(Path(f'{self.interim_dir}/{self.data_lbls[type_idx]}').iterdir()):
                self.data_to_im(
                    files, self.data_lbls[type_idx],
                    delimiter=self.config['delimiter']
                )

        used_data = self.config.get('used_data', 1.0)
        test_size = self.config.get('test_size', 0.2)
        train_files, test_files = train_test_split(
            np.transpose(filenames),
            test_size=1 - used_data,
            random_state=self.config.get('random_seed', 1234)
        )
        __, test_files = train_test_split(
            test_files,
            test_size=len(np.transpose(filenames)) * test_size / len(test_files),
            random_state=self.config.get('random_seed', 1234)
        )

        # self.depth = dset.ImageFolder(root='{}/{}'.format(self.interim_dir, self.target_lbl))
        # self.target = dset.ImageFolder(root='{}/{}'.format(self.interim_dir, self.target_lbl))

    def __len__(self):
        # Assume that features and targets have equal length
        return len(self.depth)

    def __getitem__(self, index):
        grain, y = self.data_grain[index]
        grain = np.asarray(grain)
        mask, _ = self.data_mask[index]
        mask=np.asarray(mask)
        mask=mask/255
        mask = mask.astype('uint8')
        # mask = onehot(mask, 2)
        mask = mask.swapaxes(0, 2).swapaxes(1, 2)
        mask = torch.FloatTensor(mask)



        #mask, _ = self.data_mask[index]
        #mask = np.array(mask, dtype=np.int32)
        #mask[mask==255]=1


        # grain=transform(grain)
        item = {'G':grain, 'M':mask}#grain(512,512,3), mask(2,512,512)
        return item

    def data_to_im(self, filenames, type_label='data', delimiter=' '):
        """Method for reading grain measurements"""
        for file_idx in tqdm(range(len(filenames))):
            filename = filenames[file_idx]
            # Identify if file contains feature or target data based on filename
            data_type = os.path.splitext(filename)[0].split('_')[-1]
            # Scale data between [0, 1]
            data = np.loadtxt(f'{self.data_dir}/{filename}', delimiter=delimiter)
            data = (data - np.min(data)) / np.ptp(data)

            # Use viridis color scale
            viridis = mpl.cm.get_cmap('gray')
            # image = viridis(data)

            desired_image_dimensions = self.config.get('processed_dim', [512, 512])
            height = desired_image_dimensions[0]
            width = desired_image_dimensions[1]

            for jdx in range(int(np.shape(data)[0] / height)):
                for idx in range(int(np.shape(data)[1] / width)):
                    sub_image = data[
                        height * jdx: height * (jdx+1),
                        width * idx: width * (idx+1)
                    ]
                    im_to_save = Image.fromarray(np.uint8(viridis(sub_image) * 255))
                    im_to_save.save(
                        '{}/{}/{}_{:03d}_xi{:d}_yi{:d}.png'.format(
                            self.interim_dir,
                            type_label,
                            data_type,
                            file_idx+1,
                            idx,
                            jdx
                        )
                    )
