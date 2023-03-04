"""Main script for running the segmentation learning task"""

import shutil
import misc

from pytorchutils.fcn8s import FCNModel as VGGModel
# from pytorchutils.fcn_resnet import FCNModel as ResNet
from graindetection.dataprocessor import DataProcessor
from graindetection.trainer import Trainer

from config import (
    DATA_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    INFER_DIR,
    DATA_LABELS,
    PROCESSED_DIR,
    data_config,
    model_config,
    INFER_DIR
)


def main():
    """Main method"""

    # shutil.rmtree(PROCESSED_DIR, ignore_errors=True)
    misc.gen_dirs(
        [DATA_DIR, RESULTS_DIR, MODELS_DIR] +
        [f'{INFER_DIR}/test/{data_lbl}/1' for data_lbl in DATA_LABELS] +
        [f'{PROCESSED_DIR}/train/{data_lbl}/1' for data_lbl in DATA_LABELS] +
        [f'{PROCESSED_DIR}/test/{data_lbl}/1' for data_lbl in DATA_LABELS]
    )

    data_processor = DataProcessor(data_config)
    model = VGGModel(model_config)
    trainer = Trainer(model_config, model, data_processor)
    trainer.get_batches_fn = data_processor.get_batches
    # acc = trainer.validate(101)
    # print(f"Validation accuracy: {acc}")
    trainer.train(validate_every=5, save_every=5)
    # trainer.infer(INFER_DIR)

if __name__ == '__main__':
    misc.to_local_dir(__file__)
    main()
