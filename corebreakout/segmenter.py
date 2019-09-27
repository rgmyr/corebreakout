"""
Raw core image processing using Mask-RCNN model(s).

Mask-RCNN implementation from `mrcnn` package: github/matterport/Mask_RCNN

TODO:
    - allow other orientations + object classes in `segment` method
"""
from pathlib import Path
from operator import add
from functools import reduce
from math import ceil

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology, measure

import mrcnn.model as modellib
from mrcnn.utils import Dataset as mrcnn_Dataset

from corebreakout import CoreColumn
from corebreakout import mrcnn_model
from corebreakout import defaults, datasets, layout, utils

# There seem to be two slightly different offsets
# Eventually we should get away from this, maybe by also detecting core box boundaries

endpts = {
    'A': (815, 6775),
    'B': (325, 6600)
}


class CoreSegmenter:
    """Mask-RCNN model container for extraction and aggregation of columns from core images.

    `model_dir` and `weights_path` must be passed to constructor.
    """
    def __init__(self, model_dir, weights_path, model_config=defaults.DefaultConfig, class_names=defaults.CLASSES, layout_params={}):
        """
        Parameters
        ----------
        model_dir : str or Path
            Path to saved MRCNN model directory
        weights_path : str or Path
            Path to saved weights file of corresponding model
        model_config : mrcnn.Config, optional
            MRCNN configuration object, default=`corebreakout.defaults.DefaultConfig`.
        dataset : mrcnn.utils.Dataset, optional
            An instance of `Dataset` or subclass to use for class names, etc. If None,
            will attempt to use a `PolygonDataset` with default parameters.
        layout_params : dict, optional
            Any layout parameters to override from default=`corebreakout.defaults.LAYOUT_PARAMS`.
            See `defaults.py` for explanations and options for each parameter.
        """
        self.model_config = model_config

        # Check that `class_names` make sense
        if len(class_names) == (model_config.NUM_CLASSES - 1):
            self.class_names = ['BG'] + class_names
        else:
            assert len(class_names) == model_config.NUM_CLASSES, 'Number of `class_names` must match `model_config`'
            assert class_names[0] == 'BG', 'Background `BG` must be first in `class_names`'
            self.class_names = class_names

        # Set defaults and check validity of new params via setter
        self._layout_params = defaults.LAYOUT_PARAMS
        self.layout_params = layout_params

        # Build and load the saved model
        print(f'Building MRCNN model from directory: {str(model_dir)}')
        self.model = modellib.MaskRCNN(mode='inference',
                                      config=self.model_config,
                                      model_dir=model_dir)

        print(f'Loading model weights from file: {str(weights_path)}')
        self.model.load_weights(weights_path, by_name=True)

    @property
    def layout_params(self):
        return self._layout_params

    @layout_params.setter
    def layout_params(self, new_params):
        self._layout_params.update(new_params)
        self._check_layout_params()
        self.column_class_id = self.class_names.index(self.layout_params['col_class'])
        if self.endpts_is_class:
            self.endpts_class_id = self.class_names.index(self.layout_params['endpts'])


    def segment(self, img, depth_range, add_tol=None, add_mode='fill', layout_params={}, show=False):
        """
        Detect and segment core columns in `img`, return stacked CoreColumn instance.

        Parameters
        ----------
        img : str or array
            Filename or RGB image array to segment
        depth_range : list(float)
            Top and bottom depths of set of columns in image
        add_tol : float, optional
            Tolerance for adding discontinuous columns. Default=None results in tolerance ~ image resolution.
        add_mode : one of {'fill', 'collapse'}, optional
            Add mode for generated `CoreColumn` instances (see `CoreColumn` docs)
        layout_params : dict, optional
            Any layout parameters to override.
        show : boolean, optional
            Set to True to show images/masks at each step

        Returns
        -------
        img_col : CoreColumn
            Stacked/aggregated `CoreColumn` object
        """
        # Note: assignment calls setter to update, checks validity
        self.layout_params = layout_params

        # Is `depth_range` sane?
        if min(depth_range) == 0.0 or depth_range[1]-depth_range[0] == 0.0:
            raise UserWarning(f'depth_range {depth_range} is suspect... make you are passing valid depths.')

        # If `img` points to file, read it. Otherwise assumed to be valid array.
        if isinstance(img, (str, Path)):
            print(f'Reading file: {img}')
            img = io.imread(img)

        # Set up expected number of columns and their individual depths
        col_height = layout_args['col_height']
        num_expected = ceil(depth_range[1]-depth_range[0] / col_height)
        col_tops = [depth_range[0]+i*col_height for i in range(num_expected)]
        col_bots = [top+col_height for top in col_tops]

        # Get MRCNN column predictions
        preds = self.model.detect([img], verbose=0)[0]
        if show:
            utils.show_preds(img, preds, self.class_names)

        # Select masks for column class, convert to 2D labels array
        col_masks = preds['masks'][:,:,preds['class_ids']==self.column_class_id]
        col_labels = utils.masks_to_labels(col_masks)

        # Get sorted `skimage` regions for column masks
        col_regions = layout.sort_regions(measure.regionprops(col_labels), self.layout_params['order'])

        # Figure out crop endpoints, set related args
        crop_axis = 0 if self.layout_params['orientation'] == 'l2r' else 1

        if self.endpts_is_class:
            measure_idxs = np.where(preds['class_ids'] == self.endpts_class_id)
            # If object not detected, then ignore for cropping
            if measure_idxs.size == 0:
                crop_axis, endpts = None, None
            # Otherwise, use bbox of instance with highest confidence score
            else:
                best_idx = measure_idxs[np.argmax(preds['scores'][measure_idxs])]
                measure_bbox = measure.regionprops(preds['masks'][:,:,best_idx])[0].bbox
                if crop_axis is 0:
                    endpts = (measure_bbox[1], mesaure_bbox[3])
                else:
                    endpts = (measure_bbox[0], measure_bbox[2])

        # Set single argument lambda functions to apply to column regions / region images
        crop_fn = lambda region: layout.crop_region(img, col_labels, region, axis=crop_axis, endpts=endpts)
        transform_fn = lambda region: layout.transform_region(region, self.layout_params['orientation'])

        # Apply cropping and transform (rotation) to column regions
        crops = [transform_fn(crop_fn(region)) for region in col_regions]

        # Assemble `CoreColumn` objects from masked/cropped image regions
        cols = [CoreColumn(crop, top=t, base=b,
                          add_tol=add_tol,
                          add_mode=add_mode) for crop, t, b in zip(crops, col_tops, col_bots)]

        # Slice the bottom depth if necessary
        cols[-1] = cols[-1].slice_depth(base=depth_range[1])

        # Return the concatenation of all column objects
        return reduce(add, cols)


    def _check_layout_params(self):
        """Make sure all values in `self.layout_params` are valid, set related boolean attributes."""
        lp = self.layout_params

        # Check `order` and `orientation` validity
        assert lp['order'] in layout.ORIENTATIONS, f'{lp['order']} not a valid layout `order`.'
        assert lp['orientation'] in layout.ORIENTATIONS, f'{lp['orientation']} not a valid layout `orientation`.'
        assert lp['order'] != lp['orientation'], 'layout `order` and `orientation` cannot be the same.'

        # Check that column class exists in provided class names
        assert lp['col_class'] in self.class_names, '`col_class` must be present in `class_names`'

        # Check `endpts` validity; may be name of class or 2-tuple of endpts
        endpts = lp['endpts']
        if type(endpts) is str:
            assert endpts in self.class_names, f'{endpts} is not in {self.class_names}.'
            self.endpts_is_class, self.endpts_is_coords = True, False
        elif type(endpts) is tuple:
            assert len(endpts) == 2, f'explicit `endpts` must have length == 2, not {len(endpts)}'
            self.endpts_is_class, self.endpts_is_coords = False, True
        else:
            raise TypeError(f'`endpts` must be a class name or 2-tuple, not {type(endpts)}')
