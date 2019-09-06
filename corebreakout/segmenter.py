"""
Raw core image processing using Mask-RCNN model(s).

Mask-RCNN implementation from `mrcnn` package: github/matterport/Mask_RCNN

TODO:
    - allow other orientations + object classes in `segment` method
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology, measure

from functools import reduce
from operator import add, mul
from math import ceil

import mrcnn.model as modellib
from mrcnn.visualize import display_instances

from corebreakout import CoreColumn
from corebreakout import mrcnn_model
from corebreakout import utils, defaults, layout

# There seem to be two slightly different offsets
endpts = {
    'A': (815, 6775),
    'B': (325, 6600)
}
# Eventually we should get away from this, maybe by also detecting core box boundaries


class CoreSegmenter:
    """
    Mask-RCNN model container for extraction and stacking of columns from core images.

    `model_dir` and `weights_path` must be passed to constructor.

    Parameters
    ----------
    model_dir : str or Path
        Path to saved MRCNN model directory
    weights_path : str or Path
        Path to saved weights file of corresponding model
    model_config : mrcnn.Config, optional
        MRCNN configuration object, default=`corebreakout.defaults.DefaultConfig`.
    """
    def __init__(self, model_dir, weights_path, model_config=defaults.DefaultConfig, classes=defaults.CLASSES, layout_params={}):

        self.model_config = model_config

        self.layout_params = {**defaults.LAYOUT_PARAMS, **layout_params}

        print(f'Building MRCNN model from directory: {str(model_dir)}')
        self.model = modellib.MaskRCNN(mode='inference',
                                      config=self.model_config,
                                      model_dir=model_dir)

        print(f'Loading model weights from file: {str(weights_path)}')
        self.model.load_weights(weights_path, by_name=True)


    def segment(self, img, depth_range, col_height=1.0, add_tol=None, add_mode='fill', layout_args=defaults.LAYOUT_PARAMS, show=False):
        """
        Detect and segment core columns in `img`, return stacked CoreColumn instance.

        TODO: add 'column_class' and 'measure_class' args (for multi-class datasets)?

        Parameters
        ----------
        img : str or array
            Filename or RGB image array to segment
        depth_range : list(float)
            Top and bottom depths of set of columns in image
        col_height : float, optional
            Expected height of a full column in same units as `depth_range`, default=1.0.
        add_tol : float, optional
            Tolerance for adding discontinuous columns. Default=None results in tolerance ~ image resolution.
        add_mode : one of {'fill', 'collapse'}, optional
            Add mode for generated CoreColumn instances (see CoreColumn docs)
            For columns with a gap <= add_tol, an empty column will be inserted between them.
        layout : char, one of {'A', 'B'}, optional
            Character specifying which "layout" to assume. Key in dictionary containing
            tuples of (L, R) column endpts as values, default='A'.
        show : boolean, optional
            Set to True to show images/masks at each step

        Returns
        -------
        img_col : CoreColumn
            Stacked/aggregated CoreColumn object
        """
        if min(depth_range) == 0.0 or depth_range[1]-depth_range[0] == 0.0:
            raise UserWarning(f'depth_range {depth_range} is suspect... make you are passing valid depths.')

        if isinstance(img, (str, Path)):
            print(f'Reading file: {img}')
            img = io.imread(img)

        # Set up expected num_cols and depths
        col_height = layout_args['col_height']
        num_expected = ceil(depth_range[1]-depth_range[0] / col_height)
        col_tops = [depth_range[0]+i*col_height for i in range(num_expected)]
        col_bots = [top+col_height for top in col_tops]

        # Get MRCNN column predictions
        preds = self.model.detect([img], verbose=0)[0]
        if show:
            # TODO: call without `Dataset` object?
            # ... need class_ids and label names somehow
            utils.display_preds(img, preds['rois'], preds['masks'], preds['class_ids'],
                             ['BG', 'core_column'], preds['scores'], figsize=(15,15))

        # TODO: deal with overlapping/seperated single columns?
        # Also need to differntiate `column`s from other classes
        labels = utils.masks_to_labels(preds['masks'])

        #regions = layout.sort_regions(measure.regionprops(labels), 't2b')

        # TODO: support for other orientations
        regions.sort(key=lambda x: x.bbox[0])

        crops = [np.rot90(utils.region_crop(img, labels, region, endpts[layout]), k=3) for region in regions]

        cols = [CoreColumn(crop, top=t, base=b,
                          add_tol=add_tol,
                          add_mode=add_mode) for crop, t, b in zip(crops, col_tops, col_bots)]

        # Slice the bottom depth if necessary
        cols[-1] = cols[-1].slice_depth(base=depth_range[1])

        img_col = reduce(add, cols)

        return img_col
