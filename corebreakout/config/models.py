"""
Configuration of best models for current usage.
"""
import os

# Current best Mask RCNN model
mrcnn_model_dir = '/home/'+os.environ['USER']+'/Dropbox/core_data/saved_models/mrcnn/latest'
# Original:
# mrcnn_weights_path = os.path.join(mrcnn_model_dir, 'core20180801T1515/mask_rcnn_core_0066.h5')
# New:
mrcnn_weights_path = os.path.join(mrcnn_model_dir, 'core20180920T1528/mask_rcnn_core_0024.h5')

def current_best_mrcnn():
    '''Return loaded instance of best Mask RCNN.'''
    pass

# Current best Facies prediction model
# <INFO HERE>
