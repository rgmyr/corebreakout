"""
Specify default relative paths + Mask_RCNN model config.
"""
import os
from pathlib import Path

from mrcnn.config import Config

from corebreakout import __file__ as PKG_FILE


####+++++++++++++++####
#### Default Paths #### (relative to package source)
####+++++++++++++++####

INCLUDE_DIR = Path(PKG_FILE).parent.parent / 'include'
DATASET_DIR = INCLUDE_DIR / 'pretrain/data'


MODEL_DIR = Path('/home/'+os.environ['USER']+'/Dropbox/models/corebreakout')
COCO_MODEL_PATH = MODEL_DIR / 'mask_rcnn_coco.h5'

# Current Mask RCNN experiments
TRAIN_DIR = MODEL_DIR / 'pretrain'


#weights_path = model_dir / 'core20180920T1528/mask_rcnn_core_0024.h5'
# Original:
# mrcnn_weights_path = os.path.join(mrcnn_model_dir, 'core20180801T1515/mask_rcnn_core_0066.h5')


####++++++++++++++++++++++++####
#### Default Dataset Params ####
####++++++++++++++++++++++++####

DEFAULT_CLASSES = ['col', 'tray']

# Should `Layout` be a class?
DEFAULT_LAYOUT = {
    'sort_axis' : 1,        # columns laid out vertically, ordered horizontally (0 for the inverse)
    'sort_order' : +1,      # +1 for left-to-right or top-to-bottom (-1 for right-to-left or bottom to top)
    'col_height' : 1.0,
    'endpts' : (815, 6775)  # can also be name of a class for object-based column endpoints
}


class DefaultConfig(Config):
    """
    Override some default Mask_RCNN `Config` values.

    See: https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py
    """
    NAME = 'cb_default'

    # TODO: STD_DEVs?

    # Config model head
    NUM_CLASSES = 1 + len(DEFAULT_CLASSES)
    BACKBONE = 'resnet101'
    RPN_ANCHOR_SCALES = (64, 128, 192, 320, 352)
    RPN_ANCHOR_RATIOS = [0.1, 0.5, 1]
    RPN_NMS_THRESHOLD = 0.7

    # May need to increase?
    DETECTION_MAX_INSTANCES = 6
    DETECTION_MIN_CONFIDENCE = 0.95

    # Number of train/test images, respectively
    STEPS_PER_EPOCH = 76
    VALIDATION_STEPS = 5

    # Conservative batch size. Assumes single GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
