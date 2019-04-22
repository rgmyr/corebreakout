"""
Configuration of best models for current usage.
"""
import os
from pathlib import Path


DEFAULT_CLASSES = ['col', 'box', 'scale']

DEFAULT_LAYOUT = {
    'sort_axis' : 1,        # columns laid out vertically, ordered horizontally (0 for the inverse)
    'sort_order' : +1,      # +1 for left-to-right or top-to-bottom (-1 for right-to-left or bottom to top)
    'endpts' : (815, 6775)  # can also be name of a class for object-based column endpoints
}


# Default paths (relative to package source)
INCLUDE_DIR = Path(corebreakout.__file__).parent.parent / 'include'

COCO_MODEL_DIR = INCLUDE_DIR / 'pretrained/models'
COCO_MODEL_PATH = COCO_MODEL_DIR / 'mask_rcnn_coco.h5'

DATASET_DIR = INCLUDE_DIR / 'pretrained/data'



# Current best Mask RCNN model
model_dir = Path('/home/'+os.environ['USER']+'/Dropbox/core_data/saved_models/mrcnn/latest')

weights_path = model_dir / 'core20180920T1528/mask_rcnn_core_0024.h5'

# Original:
# mrcnn_weights_path = os.path.join(mrcnn_model_dir, 'core20180801T1515/mask_rcnn_core_0066.h5')
# New:

def current_best_mrcnn():
    '''Return loaded instance of best Mask RCNN.'''
    pass
