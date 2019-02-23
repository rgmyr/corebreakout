"""
Script for processing pair(s) of picks.csv + depth.npy files to generate row-wise labels.npy.

Run with --help argument to see usage. Set COLUMNS and DTYPE to use by editing params in script.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Set columns to use from PICKS csv
TOP_COL = 'top'
BASE_COL = 'base'
LABEL_COL = 'lithology'

# Set `dtype` for labels array
# 'a2' used for 2-char strings
LABEL_DTYPE = 'a2'


def common_path(path_list):
    """
    Return common prefix of all paths in `path_list` (as a str).
    """
    chars = []
    for tup in zip(*[str(p) for p in path_list]):
        if len(set(tup)) == 1:
            chars.append(tup[0])
        else:
            break
    return ''.join(chars)


def picks_to_rows(picks_file, depth_file):
    """
    Take a picks and depth file, save the row-wise labels .npy file.
    """
    picks = pd.read_csv(picks_file, usecols=[TOP_COL, BASE_COL, LABEL_COL])

    depth = np.load(depth_file)
    row_labels = np.zeros_like(depth, dtype=LABEL_DTYPE)

    current_idx = 0
    current_pick = picks.iloc[idx]

    for i in range(row_labels.size):

        if depth[i] > current_pick[BASE_COL]:
            current_idx += 1
            current_pick = picks.iloc[current_idx]

        row_labels[i] = current_pick[LABEL_COL]

    save_path = common_path([picks_file, depth_file])+'labels.npy'

    print('Saving to:', save_path)
    np.save(save_path, row_labels)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('dir', type=str,
        help="Path to directory containing target *_depth.npy and *_picks.csv files.")

    parser.add_argument('--wells_prefix', type=str, default='',
        help="Common prefix of target *_depth.npy and *_picks.csv files, default=''.")

    args = parser.parse_args()

    picks_path = Path(args.dir) / args.well_prefix
    depths_path = Path(args.dir) / args.well_prefix

    picks_files = sorted(picks_path.glob('*_picks.npy'))
    depths_files = sorted(depths_path.glob('*_depth.npy'))

    assert len(picks_files) > 0, 'Must be at least one pair of picks + depth files'
    assert len(picks_files) == len(depth_files), 'Must be same # of picks and depths files'

    for p_file, d_file in zip(picks_files, depth_files):
        print('\nGenerating row labels for pair of files:\n', p_file.name, d_file.name)
        picks_to_rows(p_file, d_file)

