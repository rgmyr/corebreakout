"""
Script for processing batch of raw images in a directory into saved `CoreColumn`s.

The `path` given should contain images as jpeg files, and a `depth_csv`.csv file in the format:

```
           ,    top,    bottom
<filename1>, <top1>, <bottom1>
...
<filenameN>, <topN>, <bottomN>
```

NOTE: model `Config`, `class_names`, and segmentation `layout_params` can only be
changed manually at the top of script.

Run with --help argument to see full options.
"""
import os
import argparse
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import reduce
from operator import add

from corebreakout import defaults
from corebreakout import CoreSegmenter, CoreColumn

# Change Config selection manually
model_config = defaults.DefaultConfig()

# Change class_names manually
class_names = defaults.CLASSES

# Change any non-default layout_params manually
layout_params = defaults.LAYOUT_PARAMS


parser = argparse.ArgumentParser(description='Convert image directories to saved CoreColumns.')
parser.add_argument('path',
    type=str,
    help="Path to directory of images (and depth information csv) to process."
)
parser.add_argument('--model_dir',
    type=str,
    default=defaults.MODEL_DIR,
    help="Directory to load `mrcnn` model from. Default=defaults.MODEL_DIR"
)
parser.add_argument('--weights_path',
    type=str,
    default=defaults.CB_MODEL_PATH,
    help="Path to model weights to load. Default=defaults.CB_MODEL_PATH"
)
parser.add_argument('--add_tol',
    dest='add_tol',
    type=float,
    default=5.0,
    help="Gap tolerance when adding CoreColumn objects, default=5.0."
)
parser.add_argument('--add_mode',
    dest='add_mode',
    default='fill',
    help="CoreColumn.add_mode. One of {\'fill\', \'collapse\'}."
)
parser.add_argument('--depth_csv',
    dest='depth_csv',
    type=str,
    default='auto_depths.csv',
    help="Name of filename + (top, bottom) csv to read from `path`, default=\'auto_depths.csv\'"
)
parser.add_argument('--save_dir',
    dest='save_dir',
    type=str,
    default=None,
    help="Path to save CoreColumn to, default=None will save to `path`"
)
parser.add_argument('--save_name',
    dest='save_name',
    default=None,
    help='Name to use for `CoreColumn.save`, default=None results in \'CoreColumn_<top>_<base>\''
)
parser.add_argument('--save_mode',
    dest='save_mode',
    type=str,
    default='pickle',
    help='One of {\'pickle\', \'numpy\'}. Whether to save as single `pkl` or multiple `npy` files'
)


def main():
    args = parser.parse_args()

    # Check path exists
    print('Reading from path:', args.path)
    assert os.path.exists(args.path), f'{args.path} does not exist.'

    # Get file paths
    img_paths = sorted(glob(os.path.join(args.path,'*.jpeg')))
    assert len(img_paths) != 0, '`path` must contain at least one jpeg image'

    # Read image depths
    depths_df = pd.read_csv(os.path.join(args.path, args.depth_csv), index_col=0)
    run_img_paths = [p for p in img_paths if p.split('/')[-1] in depths_df.index]
    tops = depths_df.top.values.astype(float)
    bottoms = depths_df.bottom.values.astype(float)

    # Sanity check : csv vs dir contents
    if len(tops) != len(run_img_paths):
        img_path_names = [p.split('/')[-1] for p in run_img_paths]
        raise ValueError(f'Files in csv but not directory: {set(depths.index)-set(img_path_names)}')

    # Sanity check : maximum gap b/t images
    depth_gaps = (tops[1:] - bottoms[:-1])
    maximum_gap = depth_gaps.max()
    if maximum_gap > args.add_tol:
        img_gap_loc = run_img_paths[depth_gaps.argmax()]
        raise ValueError(f'Maximum gap of {maximum_gap} at {img_gap_loc} exceeds {args.add_tol}.')

    # Segment images
    segmenter = CoreSegmenter(
                args.model_dir,
                args.weights_path,
                model_config=model_config,
                class_names=class_names,
                layout_params=layout_params
    )

    segment = lambda f, t, b : segmenter.segment(f, [t, b], add_tol=args.add_tol, add_mode=args.add_mode)

    cols = [segment(f, t, b) for f, t, b in zip(run_img_paths, tops, bottoms)]

    full_column = reduce(add, cols)

    print(f'Created CoreColumn with depth_range={full_column.depth_range}')

    # Save the CoreColumn
    save_dir = args.save_dir or args.path
    print(f'Saving CoreColumn to {save_dir} in mode {args.save_mode}')

    if args.save_mode is 'pickle':
        full_column.save(save_dir, name=args.save_name, pickle=True)
    else:
        full_column.save(save_dir, name=args.save_name, pickle=False, image=True, depths=True)

if __name__ == '__main__':
    main()
