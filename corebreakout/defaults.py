"""
Default paths, dataset and Mask_RCNN model config, `CoreColumn` viz settings.
"""
import os
from pathlib import Path

from mrcnn.config import Config

from corebreakout import __file__ as PKG_FILE


####+++++++++++++++####
#### Default Paths ####
####+++++++++++++++####

ASSETS_DIR = Path(PKG_FILE).parent.parent / "assets"

# Where to find train/test subdirectories
DATASET_DIR = ASSETS_DIR / "data"

# Where to find saved model weights
MODEL_DIR = ASSETS_DIR / "models"
# Default "core" model weights
CB_MODEL_PATH = MODEL_DIR / 'mask_rcnn_cb_default.h5'
# Pretrained COCO model weights
COCO_MODEL_PATH = MODEL_DIR / 'mask_rcnn_coco.h5'

# Where to save Mask RCNN training checkpoints, etc.
TRAIN_DIR = MODEL_DIR


####++++++++++++++++++++++++####
#### Default Dataset Params ####
####++++++++++++++++++++++++####

CLASSES = ["col", "tray"]

# See `docs/layout_parameters.md` for more information
LAYOUT_PARAMS = {
    "order": "t2b",  # depth order by which to sort set of columns
    "orientation": "l2r",  # depth orientation of each individual column
    "col_height": 1.0,  # assumed height of each column, or tray, etc.
    "col_class": "col",  # name of class for core sample columns
    "endpts": "tray",  # name of class, 'auto', 'auto_all', or 2-tuple
}

####++++++++++++++++++++++####
#### Default Model Config ####
####++++++++++++++++++++++####


class DefaultConfig(Config):
    """M-RCNN model configuration.

    Override some default Mask_RCNN `Config` values.

    For all available parameters and explanations, see:
        https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py
    """

    NAME = "cb_default"

    # Number of classes, including background
    NUM_CLASSES = 1 + len(CLASSES)

    BACKBONE = "resnet101"

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (64, 128, 192, 256, 320)

    # < 1 : wide anchor, > 1 : tall anchor
    # These defaults assume horizontal (wide) columns
    # Note: starting from COCO model requires exactly 3 anchor ratios
    RPN_ANCHOR_RATIOS = [0.2, 0.5, 1]

    # Non-max suppresion threshold. Increasing generates more proposals.
    RPN_NMS_THRESHOLD = 0.9  # default = 0.7

    # STD_DEVs? Probably not, shouldn't make a big difference.

    # Maximum number of detections and minimum confidence
    DETECTION_MAX_INSTANCES = 6
    DETECTION_MIN_CONFIDENCE = 0.98

    # Set to default number of train/test images, respectively
    # (Can increase former to validate less often though)
    STEPS_PER_EPOCH = 25
    VALIDATION_STEPS = 5

    # Modify loss weights for more precise optimization
    # Few classes present, so we can lower `class` losses
    LOSS_WEIGHTS = {
        "rpn_class_loss": 0.5,
        "rpn_bbox_loss": 1.0,
        "mrcnn_class_loss": 0.5,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 1.0,
    }

    # Conservative batch size + assuming single GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


####++++++++++++++++++++++++++++++++####
#### Default CoreColumn plot params ####
####++++++++++++++++++++++++++++++++####
"""Set the default parameters for (fig, ax) returned by `CoreColumn.plot()`.

DEPTH_TICK_ARGS -- passed to `viz.make_depth_ticks()`
MAJOR_TICK_PARAMS -- passed to `ax.tick_params(which='major', ...)`
MINOR_TICK_PARAMS -- passed to `ax.tick_params(which='minor', ...)`

You can also add additional arguments to both `*_TICK_PARAMS` from:
    https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.tick_params.html
"""
DEPTH_TICK_ARGS = {
    "major_precision": 0.1,
    "major_format_str": "{:.1f}",
    "minor_precision": 0.01,
    "minor_format_str": "{:.2f}",
}

MAJOR_TICK_PARAMS = {
    "labelleft": True,  # False to disable labels
    "labelsize": 32,
    "labelcolor": "black",
    "left": True,  # False to disable ticks
    "length": 35,
    "width": 4,
    "color": "black",
}

MINOR_TICK_PARAMS = {
    "labelleft": True,  # False to disable labels
    "labelsize": 12,
    "labelcolor": "black",
    "left": True,  # False to disable ticks
    "length": 5,
    "width": 4,
    "color": "black",
}
