"""Script for evaluating already trained models with different checkpoints"""

import re
from glob import glob
import numpy as np
import misc

from pytorchutils.globals import torch, nn
# from pytorchutils.fcn8s import FCNModel
from pytorchutils.fcn_resnet import FCNModel
from pytorchutils.ahg import AHGModel
from graindetection.dataprocessor import DataProcessor
from graindetection.trainer import Trainer

from config import data_config, model_config, MODELS_DIR


def main():
    """Main method"""

    data_processor = DataProcessor(data_config)

    total_dirs = glob(f'{MODELS_DIR}/3fullattn/')
    # total_dirs.sort(key=lambda f: int(re.search(r'\d+', f.split('_')[-1]).group()))

    for directory in total_dirs:
        print(f"Processing model directory: {directory}")
        # n_measurements = int(re.search(r'\d+', directory.split('_')[-1]).group())
        checkpoints = glob(f'{directory}*DataParallel*')
        checkpoints.sort(key=lambda f: int(re.search(r'\d+', f.split('_')[-2]).group()))

        accuracies = np.zeros((len(checkpoints), 3))
        stds = np.zeros((len(checkpoints), 3))
        for idx, checkpoint in enumerate(checkpoints):
            epoch = int(re.search(r'\d+', checkpoint.split('_')[-2]).group())
            model = nn.DataParallel(AHGModel(model_config))
            # model = FCNModel(model_config)
            print(f"checkpoint file: {checkpoint}")
            state = torch.load(checkpoint)
            model.load_state_dict(state['state_dict'])
            trainer = Trainer(model_config, model, data_processor)
            accuracy, std = trainer.validate(epoch)
            accuracies[idx] = accuracy
            stds[idx] = std
            print(
                f'epoch: {epoch}, iou: {accuracy[0]:.3f} +- {std[0]:.3f}, '
                f'pacc: {accuracy[1]:.2f} +- {std[1]:.2f}, '
                f'bpacc: {accuracy[2]:.2f} +- {std[2]:.2f}, '
            )
        np.save(f'{directory}accuracy_progression.npy', np.array([accuracies, stds]))

if __name__ == '__main__':
    misc.to_local_dir(__file__)
    main()
