"""
Train and save model from data in `assets/data`.
"""
import argparse
import warnings

import mrcnn.model as modellib

from corebreakout import defaults
from corebreakout.datasets import PolygonDataset

"""
parser = argparse.ArgumentParser(description="Train a new MRCNN model from COCO weights")
parser.add_argument(
    '--steps',
    type=int,
    default=3,
    help="1, 2, or 3. How many of the steps to train: (heads, 4+, entire model)"
)
parser.add_argument(
    '--save_name'
    type=str,
    default='auto_depths',
    help="Name of csv file(s) saved in image subdirectory (or subdirectories)."
)
parser.add_argument(
    '--force',
    dest='force',
    action='store_true',
    help="Flag to force overwrite of any existing <save_name>.csv files."
)
parser.add_argument(
    '--inspect',
    dest='inspect',
    action='store_true',
    help="Flag to inspect images and OCR output whenever there is an issue."
)
"""


# Select model configuration to use
model_config = defaults.DefaultConfig()

# Set model training directory
model_dir = defaults.TRAIN_DIR / 'modified_rpn'

# Set up and load train/test datasets
data_root = defaults.DATASET_DIR

train_dataset = PolygonDataset()
train_dataset.collect_annotated_images(data_root, 'train')
train_dataset.prepare()

test_dataset = PolygonDataset()
test_dataset.collect_annotated_images(data_root, 'test')
test_dataset.prepare()


# Build model in training mode, with COCO weights
model = modellib.MaskRCNN(mode="training", config=model_config,
                          model_dir=str(model_dir))

model.load_weights(str(defaults.COCO_MODEL_PATH), by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                            "mrcnn_bbox", "mrcnn_mask"])

# Three step training proces
# For the BGS dataset, seems to be diminishing returns after ~100 epochs
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # Train network heads
    print('\n\nTraining network heads')
    model.train(train_dataset, test_dataset,
                learning_rate=model_config.LEARNING_RATE,
                epochs=40,
                layers='heads')

    # Finetune layers from ResNet stage 4 and up
    print('\n\nTuning stage 4 and up')
    model.train(train_dataset, test_dataset,
                learning_rate=model_config.LEARNING_RATE / 10,
                epochs=100,
                layers='4+')

    # Finetune all layers of the model
    print('\n\nTuning all model layers')
    model.train(train_dataset, test_dataset,
                learning_rate=model_config.LEARNING_RATE / 100,
                epochs=150,
                layers='all')
