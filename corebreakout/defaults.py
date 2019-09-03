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

CLASSES = ['col', 'tray']

LAYOUT_ARGS = {
    'order' : 't2b',
    'orientation' : 'l2r',
    'col_height' : 1.0,
}


class DefaultConfig(Config):
    """
    Override some default Mask_RCNN `Config` values.

    See: https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py
    """
    NAME = 'cb_default'

    # Number of classes, including background
    NUM_CLASSES = 1 + len(CLASSES)

    BACKBONE = 'resnet101'

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (64, 128, 192, 320, 352)

    # < 1 : wide anchor, > 1 : tall anchor
    RPN_ANCHOR_RATIOS = [0.1, 0.4, 0.7, 1]

    # Non-max suppresion threshold. Increasing generates more proposals.
    RPN_NMS_THRESHOLD = 0.7

    # TODO: STD_DEVs?

    # May need to increase?
    DETECTION_MAX_INSTANCES = 6
    DETECTION_MIN_CONFIDENCE = 0.95

    # Number of train/test images, respectively
    STEPS_PER_EPOCH = 17
    VALIDATION_STEPS = 3

    # Modify loss weights for more precise optimization
    # Should we lower 'class' losses perhaps?
    LOSS_WEIGHTS = {
        "rpn_class_loss": 0.1,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 0.1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    # Conservative batch size. Assumes single GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
