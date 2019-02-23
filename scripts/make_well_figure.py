"""
Script for processing batch of images in a directory into a single figure (image+log)

Run with --help argument to see usage.
"""
import os
import pathlib
import argparse
from glob import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from functools import reduce
from operator import add

from coremdlr.coresegmenter import CoreSegmenter
from coremdlr.corecolumn import CoreColumn

from coremdlr.models_config import mrcnn_model_dir, mrcnn_weights_path


default_base_path = '/home/'+os.environ['USER']+'/Dropbox/core_data/well_core_data/wos_data/schiehallion-complex/'


def load_segmenter():
    return CoreSegmenter(mrcnn_model_dir, mrcnn_weights_path)


def is_good_dir(d, save_ext, force_overwrite):
    if not os.path.exists(d):
        print(d, ' does not exist')
        return False
    no_existing_file = len(glob(os.path.join(d,'*'+save_ext))) == 0
    if not (force_overwrite or no_existing_file):
        return False
    else:
        return True

def make_dir_figure(args):
    conv_dir = os.path.join(default_base_path, args.well, 'converted')
    print('Reading from well:', conv_dir)

    if not is_good_dir(conv_dir, args.save_ext, args.force):
        print(conv_dir, ' has existing file with args.save_ext and args.force==False... doing nothing.')
        return

    # Get file paths
    img_paths = sorted(glob(os.path.join(conv_dir,'*.jpeg')))
    assert len(img_paths) != 0, '`dir`/converted/ must contain at least one jpeg image'
    las_path = list(pathlib.PosixPath(conv_dir).parent.glob('*.las'))
    assert len(las_path) == 1, '`dir` must contain exactly one LAS file'

    # Read images + depths
    depths = pd.read_csv(os.path.join(conv_dir, args.depth_csv), index_col=0)
    print(depths)
    run_img_paths = [p for p in img_paths if p.split('/')[-1] in depths.index]
    tops = depths.top.values.astype(float)
    bottoms = depths.bottom.values.astype(float)
    assert len(tops) == len(run_img_paths), 'All images listed in depth csv must exist'
    # TODO: check for image name & csv row match

    # Segment images
    segmenter = load_segmenter()
    cols = [segmenter.segment(f, [t, b], layout=args.layout.upper(), add_tol=args.add_tol) \
            for f, t, b in zip(run_img_paths, tops, bottoms)]
    core_col = reduce(add, cols)
    print(f'Created Column with depth_range={core_col.depth_range}')

    # Create figure
    aspect = core_col.img.shape[1] / core_col.img.shape[0]
    fig, ax = plt.subplots(ncols=2, figsize=(6*aspect*args.fig_height, args.fig_height))

    core_col.plot_image(ax=ax[0])
    core_col.plot_logs(ax[1], las_file=str(las_path[0]))

    fig.savefig(os.path.join(conv_dir, args.save_ext), bbox_inches='tight')


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
        help="Which layout assumptions to use, one of [\'A\', \'B\'], default=\'A\'"
    )
    parser.add_argument('--add_tol',
        dest='add_tol',
        type=float,
        default=0.0,
        help="Gap tolerance when adding CoreColumn objects, default=0.0."
    )
    parser.add_argument('--depth_csv',
        dest='depth_csv',
        type=str,
        default='auto_depths.csv',
        help="Name of depth file to read from `converted` directory, default=\'auto_depths.csv\'"
    )
    parser.add_argument('--fig_height',
        dest='fig_height',
        type=int,
        default=600,
        help="Height of output matplotlib figure, default=600."
    )
    parser.add_argument('--save_ext',
        dest='save_ext',
        type=str,
        default='_figure.pdf',
        help="Tail of the created filenames for figures. <something>.pdf is recommended, default=\'_figure.pdf\'"
    )
    parser.add_argument('--force',
        dest='force',
        action='store_true',
        help="Flag for forcing overwrite of any existing files with the same `save_ext`"
    )

    make_dir_figure(parser.parse_args())


if __name__ == '__main__':
    main()
