"""Script for evaluating already trained models with different checkpoints"""

import re
from glob import glob
import numpy as np
import misc

from pytorchutils.globals import torch
from pytorchutils.fcn8s import FCNModel
from graindetection.dataprocessor import DataProcessor
from graindetection.trainer import Trainer

from config import data_config, model_config, MODELS_DIR


def main():
    """Main method"""

    data_processor = DataProcessor(data_config)

    total_dirs = glob(f'{MODELS_DIR}/pretrained_sweet_spot_wear26*/')
    # total_dirs.sort(key=lambda f: int(re.search(r'\d+', f.split('_')[-1]).group()))

    for directory in total_dirs:
        print(f"Processing model directory: {directory}")
        # n_measurements = int(re.search(r'\d+', directory.split('_')[-1]).group())
        checkpoints = glob(f'{directory}FCNModel*')
        print(checkpoints)
        quit()
        checkpoints.sort(key=lambda f: int(re.search(r'\d+', f.split('_')[-2]).group()))

        accuracies = np.zeros(len(checkpoints))
        stds = np.zeros(len(checkpoints))
        for idx, checkpoint in enumerate(checkpoints):
            model = FCNModel(model_config)
            state = torch.load(checkpoint)
            model.load_state_dict(state['state_dict'])
            trainer = Trainer(model_config, model, data_processor)
            accuracy, std = trainer.validate(idx)
            accuracies[idx] = accuracy
            stds[idx] = std
            print(f'idx: {idx}, acc: {accuracy} +- {std}')
        np.save(f'{directory}accuracy_progression.npy', np.array([accuracies, stds]))

if __name__ == '__main__':
    misc.to_local_dir(__file__)
    main()
