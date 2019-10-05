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

INCLUDE_DIR = Path(PKG_FILE).parent.parent / 'assets'
DATASET_DIR = INCLUDE_DIR / 'assets/data'
#INCLUDE_DIR = Path(PKG_FILE).parent.parent / 'assets'
#DATASET_DIR = Path({PKG_FILE})

# Before release: change this to 'assets/models'
MODEL_DIR = Path('/home/'+os.environ['USER']+'/Dropbox/models/corebreakout')
COCO_MODEL_PATH = MODEL_DIR / 'mask_rcnn_coco.h5'

# Current Mask RCNN experiments
TRAIN_DIR = MODEL_DIR / 'pretrain'


# weights_path = model_dir / 'core20180920T1528/mask_rcnn_core_0024.h5'
# Original:
# mrcnn_weights_path = os.path.join(mrcnn_model_dir, 'core20180801T1515/mask_rcnn_core_0066.h5')


####++++++++++++++++++++++++####
#### Default Dataset Params ####
####++++++++++++++++++++++++####

CLASSES = ['col', 'tray']

# See `docs/layout_parameters.md` for more information
LAYOUT_PARAMS = {
    'order' : 't2b',            # depth order by which to sort set of columns
    'orientation' : 'l2r',      # depth orientation of each individual column
    'col_height' : 1.0,         # assumed height of each column, or tray, etc.
    'col_class' : 'col',        # name of class for sample material columns
    'endpts' : 'tray'           # name of class, or 2-tuple of explicit pixel rows/cols
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
    NAME = 'cb_default'

    # Number of classes, including background
    NUM_CLASSES = 1 + len(CLASSES)

    BACKBONE = 'resnet101'

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (64, 128, 192, 256, 320) #, 352)

    # < 1 : wide anchor, > 1 : tall anchor
    # These defaults assume horizontal (wide) columns
    # Note: starting from COCO model requires exactly 3 anchor ratios
    RPN_ANCHOR_RATIOS = [0.1, 0.55, 1.0]

    # Non-max suppresion threshold. Increasing generates more proposals.
    RPN_NMS_THRESHOLD = 0.7 # default = 0.7

    # STD_DEVs? Probably not, shouldn't make a big difference.

    # Maximum number of detections and minimum confidence
    DETECTION_MAX_INSTANCES = 6
    DETECTION_MIN_CONFIDENCE = 0.95

    # Default number of train/test images, respectively
    STEPS_PER_EPOCH = 25
    VALIDATION_STEPS = 5

    # Modify loss weights for more precise optimization
    # Few classes present, so we can lower `class` losses
    LOSS_WEIGHTS = {
        "rpn_class_loss": 0.5,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 0.5,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    # Conservative batch size. Assumes single GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


####++++++++++++++++++++++++++++++++####
#### Default CoreColumn plot params ####
####++++++++++++++++++++++++++++++++####
"""Set the default parameters for (fig, ax) returned by `CoreColumn.plot()`.

You can also add additional arguments to both `*_TICK_PARAMS` from:
    https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.tick_params.html
"""
MAJOR_TICK_PARAMS = {
    'labelleft' : True,         # False to disable labels
    'labelsize' : 32,
    'labelcolor' : 'black',
    'left' : True,              # False to disable ticks
    'length' : 15,
    'width' : 4,
    'color' : 'black'
}

MINOR_TICK_PARAMS = {
    'labelleft' : True,         # False to disable labels
    'labelsize' : 12,
    'labelcolor' : 'black',
    'left' : True,              # False to disable ticks
    'length' : 8,
    'width' : 4,
    'color' : 'black'
}
