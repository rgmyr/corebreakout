"""
Raw core image processing using Mask-RCNN model(s).

Mask-RCNN implementation from `mrcnn` package: github/matterport/Mask_RCNN
"""
from pathlib import Path
from operator import add
from functools import reduce
from math import ceil

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology, measure

import mrcnn.model as modellib
#from mrcnn.utils import Dataset as mrcnn_Dataset

from corebreakout import CoreColumn
from corebreakout import defaults, datasets, layout, utils

# There seem to be two slightly different offsets
# Eventually we should get away from this, maybe by also detecting core box boundaries

endpts = {
    'A': (815, 6775),
    'B': (325, 6600)
}


class CoreSegmenter:
    """Mask-RCNN model container for extracting `CoreColumn`s from core images.

    `model_dir` and `weights_path` must be passed to constructor.
    """
    def __init__(self, model_dir, weights_path, model_config=defaults.DefaultConfig(), class_names=defaults.CLASSES, layout_params={}):
        """
        Parameters
        ----------
        model_dir : str or Path
            Path to saved MRCNN model directory
        weights_path : str or Path
            Path to saved weights file of corresponding model
        model_config : `mrcnn.Config`, optional
            Instance of MRCNN configuration object, default=`defaults.DefaultConfig()`.
        class_names : list(str), optional
            A list of the class names for model output. Should be in same order as in
            the `Dataset` object that model was trained on. Default=`defaults.CLASSES`
        layout_params : dict, optional
            Any layout parameters to override from default=`defaults.LAYOUT_PARAMS`.
            See `docs/layout_parameters.md` for explanations and options for each parameter.
        """
        self.model_config = model_config

        # Check that `class_names` make sense
        if len(class_names) == (model_config.NUM_CLASSES - 1):
            self.class_names = ['BG'] + class_names
        else:
            assert len(class_names) == model_config.NUM_CLASSES, 'Number of `class_names` must match `model_config`'
            assert class_names[0] == 'BG', 'Background `BG` must be first in `class_names`'
            self.class_names = class_names

        # Set defaults and check validity of any new params via setter
        self._layout_params = defaults.LAYOUT_PARAMS
        self.layout_params = layout_params

        # Build and load the saved model
        print(f'Building MRCNN model from directory: {str(model_dir)}')
        self.model = modellib.MaskRCNN(mode='inference',
                                      config=self.model_config,
                                      model_dir=str(model_dir))

        print(f'Loading model weights from file: {str(weights_path)}')
        self.model.load_weights(str(weights_path), by_name=True)


    @property
    def layout_params(self):
        return self._layout_params

    @layout_params.setter
    def layout_params(self, new_params):
        self._layout_params.update(new_params)
        self._check_layout_params()

        self.column_class_id = self.class_names.index(self.layout_params['col_class'])

        # If endpts is class, save the corresponding `id` number
        if self.endpts_is_class:
            self.endpts_class_id = self.class_names.index(self.layout_params['endpts'])


    def segment(self, img, depth_range, add_tol=None, add_mode='fill', layout_params={}, show=False, colors=None):
        """Detect and segment core columns in `img`, return single aggregated CoreColumn instance.

        Parameters
        ----------
        img : str or array
            Filename or RGB image array to segment.
        depth_range : list(float)
            Top and bottom depths of set of columns in image.
        add_tol : float, optional
            Tolerance for adding discontinuous columns. Default=None results in tolerance ~ image resolution.
        add_mode : one of {'fill', 'collapse'}, optional
            Add mode for generated `CoreColumn` instances (see `CoreColumn` docs)
        layout_params : dict, optional
            Any layout parameters to override.
        show : boolean, optional
            Set to True to show image with predictions overlayed.
        colors : list, optional
            A list of RGBA tuples, one for each in `class_names` (excluding 'BG').
            Values should be in range [0.0, 1.0], If None, uses random colors.
            Has no effect unless `show=True`.

        Returns
        -------
        img_col : CoreColumn
            Single aggregated `CoreColumn` instance
        """
        # Note: assignment calls setter to update, checks validity
        self.layout_params = layout_params

        # Is `depth_range` sane?
        if min(depth_range) == 0.0 or depth_range[1]-depth_range[0] == 0.0:
            raise UserWarning(f'`depth_range` {depth_range} starts at 0.0 or has no extent')

        # If `img` points to a file, read it. Otherwise assumed to be valid image array.
        if isinstance(img, (str, Path)):
            print(f'Reading file: {img}')
            img = io.imread(img)

        # Set up expected number of columns and their top/base depths
        col_height = self.layout_params['col_height']
        num_expected = ceil(depth_range[1]-depth_range[0] / col_height)
        col_tops = [depth_range[0]+i*col_height for i in range(num_expected)]
        col_bases = [top+col_height for top in col_tops]

        # Get MRCNN column predictions
        preds = self.model.detect([img], verbose=0)[0]
        if show:
            if colors is not None:
                assert len(colors) == (len(self.class_names)-1), 'Number of `colors` must match number of classes'
                colors = [class_colors[i-1] for i in preds['class_ids']]
            utils.show_preds(img, preds, self.class_names, colors=colors)

        # Select masks for column class
        col_masks = preds['masks'][:,:,preds['class_ids']==self.column_class_id]

        # Check that number of columns matches expectation
        num_cols = col_masks.shape[-1]
        if num_cols != num_expected:
            raise UserWarning(f'Number of detected columns {num_cols} does not match \
                             expectation of {num_expected}')

        # Convert 3D binary masks to 2D integer labels array
        col_labels = utils.masks_to_labels(col_masks)

        # Get sorted `skimage` regions for column masks
        col_regions = layout.sort_regions(measure.regionprops(col_labels), self.layout_params['order'])

        # Figure out crop endpoints, set related args
        crop_axis = 0 if self.layout_params['orientation'] == 'l2r' else 1

        # Set up `endpts` for bbox adjustment
        if self.endpts_is_auto:
            if self.layout_params['endpts'] == 'auto':
                endpts = self._get_auto_endpts(col_regions, crop_axis)
            else:
                # 'auto_all' mode
                all_regions = measure.regionprops(utils.masks_to_labels(preds['masks']))
                endpts = self._get_auto_endpts(all_regions, crop_axis)

        elif self.endpts_is_class:
            measure_idxs = np.where(preds['class_ids'] == self.endpts_class_id)[0]
            # If object not detected, then ignore for cropping
            if measure_idxs.size == 0:
                print('`endpts` class not detected, cropping will use `auto` method')
                endpts = self._get_auto_endpts(col_regions, crop_axis)
            # Otherwise, use bbox of instance with highest confidence score
            else:
                best_idx = measure_idxs[np.argmax(preds['scores'][measure_idxs])]
                measure_bbox = measure.regionprops((preds['masks'][:,:,best_idx]).astype(np.int))[0].bbox
                if crop_axis is 0:
                    endpts = (measure_bbox[1], measure_bbox[3])
                else:
                    endpts = (measure_bbox[0], measure_bbox[2])

        elif self.endpts_is_coords:
            endpts = self.layout_params['endpts']

        else:
            raise RuntimeError()

        # REMOVE AFTER MAKING DOC IMAGES
        print(f'endpoint coords: {endpts}')

        # Set single argument lambda functions to apply to column regions / region images
        crop_fn = lambda region: layout.crop_region(img, col_labels, region, axis=crop_axis, endpts=endpts)
        transform_fn = lambda region: layout.rotate_vertical(region, self.layout_params['orientation'])

        # Apply cropping and transform (rotation) to column regions
        crops = [transform_fn(crop_fn(region)) for region in col_regions]

        # Assemble `CoreColumn` objects from masked/cropped image regions
        cols = [CoreColumn(crop, top=t, base=b,
                          add_tol=add_tol,
                          add_mode=add_mode) for crop, t, b in zip(crops, col_tops, col_bases)]

        # Slice the bottom depth if necessary
        cols[-1] = cols[-1].slice_depth(base=depth_range[1])

        # Return the concatenation of all column objects
        return reduce(add, cols)


    def segment_all(imgs, depth_ranges, **kwargs):
        """Segment a set of `imgs` with known `depth_ranges`, return concatenated `CoreColumn`

        Parameters
        ----------
        imgs : Iterable of either filepaths or numpy image arrays.
        depth_ranges: Iterable of (top, base) pairs for each
        **kwargs :
            See `segment()` docstring for options.
        """
        return reduce(add, [self.segment(img, dr, **kwargs) for img, dr in zip(imgs, depth_ranges)])


    def _find_endpts(self, preds):
        """Find column endpoints using method specified by `layout_params`."""
        pass


    def _get_auto_endpts(self, regions, crop_axis):
        """Find min/max of detected `regions` masks along `crop_axis`."""
        low_idx = 0 if crop_axis == 1 else 1
        high_idx = 2 if crop_axis == 1 else 3

        low = min(r.bbox[low_idx] for r in regions)
        high = max(r.bbox[high_idx] for r in regions)

        return (low, high)


    def _check_layout_params(self):
        """Make sure all values in `self.layout_params` are valid, set related boolean attributes."""
        lp = self.layout_params

        # Check `order` and `orientation` validity
        assert lp['order'] in layout.ORIENTATIONS, f'{lp["order"]} not a valid layout `order`.'
        assert lp['orientation'] in layout.ORIENTATIONS, f'{lp["orientation"]} not a valid layout `orientation`.'
        assert lp['order'] != lp['orientation'], 'layout `order` and `orientation` cannot be the same.'

        # Check that column class exists in provided class names
        assert lp['col_class'] in self.class_names, '`col_class` must be present in `class_names`'

        # Check `endpts` validity; may be a name of class or a 2-tuple of endpts
        endpts = lp['endpts']
        if 'auto' in str(endpts):
            assert str(endpts) in ['auto', 'auto_all'], 'Invalid `endpts` auto keyword'
            self.endpts_is_auto, self.endpts_is_class, self.endpts_is_coords = True, False, False
        if type(endpts) is str:
            assert endpts in self.class_names, f'{endpts} is not `auto_*` or in {self.class_names}.'
            self.endpts_is_auto, self.endpts_is_class, self.endpts_is_coords = False, True, False
        elif type(endpts) is tuple:
            assert len(endpts) == 2, f'explicit `endpts` must have length == 2, not {len(endpts)}'
            self.endpts_is_auto, self.endpts_is_class, self.endpts_is_coords = False, False, True
        elif endpts is not None:
            raise TypeError(f'`endpts` must be class name, 2-tuple, \'auto(_all)\', or None, not {type(endpts)}')
