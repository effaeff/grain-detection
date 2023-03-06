"""Config file"""

from pytorchutils.globals import nn

DATA_DIR = '../data/01_raw/measurements'
PROCESSED_DIR = '../data/02_processed'
RESULTS_DIR = '../results'
MODELS_DIR = '../models'
INFER_DIR = '../data/03_inference'
DATA_TYPES = ['depth', 'intensity', 'target']
DATA_LABELS = ['features', 'target']

data_config = {
    'data_dir': DATA_DIR,
    'processed_dir': PROCESSED_DIR,
    'processed_dim': [128, 128],
    'results_dir': RESULTS_DIR,
    'data_types': DATA_TYPES,
    'data_labels': DATA_LABELS,
    'delimiter': '\t',
    'train_size': 8, # Number of measurements
    'colorscales': ['viridis', 'gray'], # Colorscale for each data label
    'random_sampling': False, # Specifies if sub-images should randomly be extracted
    'n_samples': 40, # Number of randomly extracted sub-images per measurement
    'orig_size': 0.02, # Portion of original data in augmented dataset
    'random_seed': 4321,
    'batch_size': 2,
    'test_size': 1 # Number of measurements
}
model_config = {
    'models_dir': MODELS_DIR,
    'n_patches': (32, 32),
    'embedding_size': 128,
    'n_channels': 2,
    'output_size': 2,
    'arch': 'vgg16',
    'init': 'xavier_uniform_',
    'init_layers': (nn.ConvTranspose2d),
    'optimizer': 'Adam',
    'loss': 'BCELoss',
    'max_iter': 51,
    'learning_rate': 1e-2,
    'optim_betas': [0.0, 0.999],
    # 'reg_lambda': 1e-5,
    'pretrained': True
}
