"""
Train and save model from data in `include/pretrained`.
"""
# Import Mask RCNN
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.visualize import display_images, display_instances
import mrcnn.model as modellib
from mrcnn.model import log

from corebreakout import defaults
from corebreakout.datasets import PolygonDataset


model_config = defaults.DefaultConfig()


# Set up and load train/test datasets
data_root = defaults.DATASET_DIR

train_dataset = PolygonDataset()
train_dataset.collect_annotated_images(data_root, 'train')
train_dataset.prepare()

test_dataset = PolygonDataset()
test_dataset.collect_annotated_images(data_root, 'test')
test_dataset.prepare()


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=model_config,
                          model_dir=str(defaults.TRAIN_DIR))

model.load_weights(str(defaults.COCO_MODEL_PATH), by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                            "mrcnn_bbox", "mrcnn_mask"])


with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # Train network heads
    print('\n\nTraining network heads')
    model.train(train_dataset, test_dataset,
                learning_rate=model_config.LEARNING_RATE,
                epochs=20,
                layers='heads')

    # Finetune layers from ResNet stage 3 and up
    print('\n\nTuning stage 3 and up')
    model.train(train_dataset, test_dataset,
                learning_rate=model_config.LEARNING_RATE / 10,
                epochs=100,
                layers='3+')

    print('\n\nTuning all model layers')
    model.train(train_dataset, test_dataset,
                learning_rate=model_config.LEARNING_RATE / 100,
                epochs=1000,
                layers='all')
