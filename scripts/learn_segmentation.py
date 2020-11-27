"""Main script for running the segmentation learning task"""
import os
import misc

from pytorchutils.fcn8s import FCNModel
from graindetection.dataprocessor import DataProcessor
from graindetection.trainer import Trainer

def main():
    """Main method"""
    data_dir = '../data/01_raw'
    interim_dir = '../data/02_interim'
    processed_dir = '../data/03_processed'
    results_dir = '../results'
    data_config = {
        'data_dir': data_dir,
        'interim_dir': interim_dir,
        'processed_dir': processed_dir,
        'processed_dim': [512, 512],
        'data_labels': ['depth', 'target'],
        'delimiter': '\t',
        'used_data': 0.6, # Percentage of used original data measurements
        'random_seed': 1234,
        'batch_size': 4,
        'test_size': 0.2
    }
    model_config = {
        'results_dir': results_dir,
        'output_size': 2,
        'arch': 'vgg16',
        'loss': 'BCELoss',
        'max_iter': 100,
        'learning_rate': 0.01,
        'pretrained': True
    }

    misc.gen_dirs([data_dir, interim_dir, processed_dir, results_dir])

    data_processor = DataProcessor(data_config)
    # model = FCNModel(model_config)
    # trainer = Trainer(model_config, model, data_processor)
    # trainer.get_batches_fn = data_processor.get_batches
    # trainer.train(validate_every=5, save_every=1)

if __name__ == '__main__':
    misc.to_local_dir(__file__)
    main()
