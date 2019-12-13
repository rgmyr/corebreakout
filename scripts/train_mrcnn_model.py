"""
Train and save model from data in `assets/data`.

Note: must change `model_config` manually if you wish to use a different `Config` subclass.
"""
import argparse
import warnings

import mrcnn.model as modellib

from corebreakout import defaults
from corebreakout.datasets import PolygonDataset


parser = argparse.ArgumentParser(description="Train a new MRCNN model from COCO weights")
parser.add_argument('--steps',
    type=int,
    default=2,
    help="1, 2, or 3. How many of the steps to train: (heads, 4+, entire model)"
)
parser.add_argument('--model_dir',
    type=str,
    default=str(defaults.MODEL_DIR),
    help="Directory in which to create new training subdirectory."
)
parser.add_argument('--data_dir',
    type=str,
    default=defaults.DATASET_DIR,
    help="Directory in which to find `train` and `test` subdirectories."
)


args = parser.parse_args()
assert args.steps in [1,2,3], 'steps must be one of 1, 2, or 3'


# Select model configuration to use
model_config = defaults.DefaultConfig()

# Collect the data
train_dataset = PolygonDataset()
train_dataset.collect_annotated_images(args.data_dir, 'train')
train_dataset.prepare()

test_dataset = PolygonDataset()
test_dataset.collect_annotated_images(args.data_dir, 'test')
test_dataset.prepare()


# Build model in training mode, with COCO weights
model = modellib.MaskRCNN(mode="training", config=model_config,
                          model_dir=args.model_dir)

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
    if args.steps >= 2:
        print('\n\nTuning stage 4 and up')
        model.train(train_dataset, test_dataset,
                    learning_rate=model_config.LEARNING_RATE / 10,
                    epochs=100,
                    layers='4+')

    # Finetune all layers of the model
    if args.steps == 3:
        print('\n\nTuning all model layers')
        model.train(train_dataset, test_dataset,
                    learning_rate=model_config.LEARNING_RATE / 100,
                    epochs=200,
                    layers='all')
