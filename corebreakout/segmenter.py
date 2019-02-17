"""
Raw image processing using Mask-RCNN model.

Mask-RCNN implementation from: github/matterport/Mask_RCNN

TODO:
    - make everything a little more flexible and robust, esp. in segment() method
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology, measure

from functools import reduce
from operator import add, mul
from math import ceil

import mrcnn.model as modellib
from mrcnn.visualize import display_instances

from corebreakout import CoreColumn
from corebreakout.models import mrcnn_model
from corebreakout.utils.viz_utils import show_images

# There seem to be two slightly different offsets
endpts = {
    'A': (815, 6775),
    'B': (325, 6600)
}
# Eventually we should get away from this, maybe by also detecting core box boundaries


#+++++++++++++++++#
# Model Utilities #
#+++++++++++++++++#

def masks_to_labels(masks):
    '''Convert boolean (H,W,N) array for N instances to integer (H,W) in range(0,N+1).'''
    labels = np.zeros(masks.shape[0:-1])
    for i in range(masks.shape[-1]):
        labels += (i+1) * masks[:,:,i].astype(int)
    return labels.astype(int)

def region_crop(img, labels, region, endpts=endpts['A']):
    '''Adjust region bbox and return cropped mask.'''
    x0, y0, x1, y1 = region.bbox
    if y0 > endpts[0]:
        y0 = endpts[0]
    if y1 < endpts[1]:
        y1 = endpts[1]
    region_img = img * np.expand_dims(labels==region.label, -1)
    return region_img[x0:x1,y0:y1,:]


class CoreSegmenter:
    """
    Class for segmenting core columns using Mask-RCNN model.

    `model_dir` and `weights_path` must be passed to constructor.

    Parameters
    ----------
    model_dir : str or Path
        Path to saved MRCNN model directory
    weights_path : str or Path
        Path to saved weights file of corresponding model
    model_config : mrcnn.Config, optional
        MRCNN configuration object, default=core_mcrnn.CoreConfig().
    """
    def __init__(self, model_dir, weights_path, model_config=None):

        self.model_config = model_config or mrcnn_model.CoreConfig()

        print('Building MRCNN model...')
        self.mrcnn = modellib.MaskRCNN(mode='inference',
                                       config=self.model_config,
                                       model_dir=model_dir)

        print('Loading model weights...')
        self.mrcnn.load_weights(weights_path, by_name=True)


    def segment(self, img, depth_range, col_height=1.0, add_tol=0.0, add_mode='fill', layout='A', show=False):
        """
        Detect and segment core columns in `img`, return stacked CoreColumn instance.

        Parameters
        ----------
        img : str or array
            Filename or RGB image array to segment
        depth_range : list(float)
            Top and bottom depths of set of columns in image
        col_height : float, optional
            Expected height of a full column in arbitrary units, default=1.0.
        add_tol : float, optional
            Tolerance for adding discontinuous columns, default=0.0. For columns
            with a gap <= add_tol, an empty column will be inserted between them.
        layout : char, one of {'A', 'B'}, optional
            Character specifying which "layout" to assume. Key in dictionary containing
            tuples of (L, R) column endpts as values, default='A'.
        show : boolean, optional
            Set to True to show images/masks at each step w/ pyplot

        Returns
        -------
        img_col : CoreColumn
            Stacked/aggregated CoreColumn object
        """
        if min(depth_range) == 0.0 or depth_range[1]-depth_range[0] == 0.0:
            print('depth_range is suspect... make sure _depths.csv or similar contains valid depths.')

        if isinstance(img, str):
            print(f'Reading file: {img}')
            img = io.imread(img)

        # Set up expected num_cols and depths
        num_expected = ceil(depth_range[1]-depth_range[0] / col_height)
        col_tops = [depth_range[0]+i*col_height for i in range(num_expected)]
        col_bots = [top+col_height for top in col_tops]

        # Get MRCNN column predictions
        preds = self.mrcnn.detect([img], verbose=0)[0]
        if show:
            display_instances(img, preds['rois'], preds['masks'], preds['class_ids'],
                              ['BG', 'core_column'], preds['scores'], figsize=(10,10))

        labels = masks_to_labels(preds['masks'])

        regions = measure.regionprops(labels)
        regions.sort(key=lambda x: x.bbox[0])

        crops = [np.rot90(region_crop(img, labels, region, endpts[layout]), k=3) for region in regions]

        cols = [CoreColumn(crop, top=t, base=b, 
                          add_tol=add_tol, 
                          add_mode=add_mode) for crop, t, b in zip(crops, col_tops, col_bots)]

        # Slice the bottom depth if necessary
        cols[-1] = cols[-1].slice_depth(base=depth_range[1])

        img_col = reduce(add, cols)

        return img_col
