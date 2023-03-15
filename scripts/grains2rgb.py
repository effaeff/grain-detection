"""Script for passing data through first layer of AHG model to get rgb channels"""

import numpy as np
from tqdm import tqdm
from glob import glob
from matplotlib import pyplot as plt
import misc

from pytorchutils.ahg import AHGModel
from pytorchutils.globals import torch, DEVICE

from graindetection.dataprocessor import DataProcessor

from config import data_config, model_config, MODELS_DIR, RESULTS_DIR

def main():
    """Main method"""
    misc.gen_dirs([f'{RESULTS_DIR}/data_rgb/train', f'{RESULTS_DIR}/data_rgb/test'])
    data_processor = DataProcessor(data_config)

    checkpoint = glob(f'{MODELS_DIR}/ahg-vgg4-sum*/best/*Model*')[0]

    model = AHGModel(model_config).to(DEVICE)
    state = torch.load(checkpoint)
    model.load_state_dict(state['state_dict'])

    train_dataset, test_dataset = data_processor.get_datasets()

    print("Transforming training dataset to RGB...")
    get_rgb(train_dataset, model, f'{RESULTS_DIR}/data_rgb/train')
    print("Transforming test dataset to RGB...")
    get_rgb(test_dataset, model, f'{RESULTS_DIR}/data_rgb/test')

def get_rgb(dataset, model, to_dir):
    """Eval model to get rgb channels of data in dataset"""
    for idx, data in enumerate(tqdm(dataset)):
        features = data['F'].to(DEVICE)
        data_rgb = model.inp2rgb(features).detach().cpu().numpy()
        plt.imsave(f'{to_dir}/{idx:03d}.jpg', np.moveaxis(data_rgb, 0, -1))

if __name__ == '__main__':
    misc.to_local_dir(__file__)
    main()
