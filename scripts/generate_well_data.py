"""
Script for processing batch of images in a directory into image + row-depth files.

Run with --help argument to see usage.
"""
import os
import argparse
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import reduce
from operator import add

from coremdlr.coresegmenter import CoreSegmenter
from coremdlr.corecolumn import CoreColumn

from coremdlr.models_config import mrcnn_model_dir, mrcnn_weights_path


default_base_path = '/home/'+os.environ['USER']+'/Dropbox/core_data/well_core_data/wos_data/schiehallion-complex/'

default_save_dir = '/home/'+os.environ['USER']+'/Dropbox/core_data/facies/train_data/'



def load_segmenter():
    return CoreSegmenter(mrcnn_model_dir, mrcnn_weights_path)


def is_good_dir(d):
    if not os.path.exists(d):
        print(d, ' does not exist')
        return False
    else:
        return True

def make_dir_data(args):
    conv_dir = os.path.join(default_base_path, args.well, 'converted')
    print('Reading from well:', conv_dir)
    if not os.path.exists(conv_dir):
        print(conv_dir, ' does not exist... doing nothing.')
        return

    # Get file paths
    img_paths = sorted(glob(os.path.join(conv_dir,'*.jpeg')))
    assert len(img_paths) != 0, '`dir`/converted/ must contain at least one jpeg image'

    # Read image depths
    depths = pd.read_csv(os.path.join(conv_dir, args.depth_csv), index_col=0)
    run_img_paths = [p for p in img_paths if p.split('/')[-1] in depths.index]
    tops = depths.top.values.astype(float)
    bottoms = depths.bottom.values.astype(float)

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
    segmenter = load_segmenter()
    cols = [segmenter.segment(f, [t, b], layout=args.layout.upper(), add_tol=args.add_tol, add_mode='collapse') \
            for f, t, b in zip(run_img_paths, tops, bottoms)]
    core_col = reduce(add, cols)
    print(f'Created Column with depth_range={core_col.depth_range}')

    # Save the concatenatted data
    save_path = os.path.join(args.save_dir, args.well)
    print('Saving image + depth arrays to ', save_path + '_*')
    np.save(save_path+'_image.npy', core_col.img)
    np.save(save_path+'_depth.npy', core_col.x_depth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('well',
        type=str,
        help="Name of (schiehallion-complex) well [change `default_base_path` if not in S-C]."
    )
    parser.add_argument('--layout',
        dest='layout',
        type=str,
        default='A',
        help="Which layout to use, one of [\'A\', \'B\'], default=\'A\'"
    )
    parser.add_argument('--add_tol',
        dest='add_tol',
        type=float,
        default=5.0,
        help="Gap tolerance when adding CoreColumn objects, default=1.0."
    )
    parser.add_argument('--depth_csv',
        dest='depth_csv',
        type=str,
        default='auto_depths.csv',
        help="Name of depth file to read from `converted` directory, default=\'auto_depths.csv\'"
    )
    parser.add_argument('--save_dir',
        dest='save_dir',
        type=str,
        default=default_save_dir,
        help="Path to save image and depth array files to, default=None (parent of `converted`)"
    )

    make_dir_data(parser.parse_args())


if __name__ == '__main__':
    main()
