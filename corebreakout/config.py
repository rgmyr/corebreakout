"""
Configuration of best models for current usage.
"""
import os
from pathlib import Path

# Current best Mask RCNN model
model_dir = Path('/home/'+os.environ['USER']+'/Dropbox/core_data/saved_models/mrcnn/latest')

weights_path = model_dir / 'core20180920T1528/mask_rcnn_core_0024.h5'

# Original:
# mrcnn_weights_path = os.path.join(mrcnn_model_dir, 'core20180801T1515/mask_rcnn_core_0066.h5')
# New:

def current_best_mrcnn():
    '''Return loaded instance of best Mask RCNN.'''
    pass

