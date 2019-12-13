import argparse
import os
import pathlib

import numpy as np
from skimage import io


parser = argparse.ArgumentParser('Split .npy files into jpegs for labeling.')
parser.add_argument('well', type=str, help='Name of well to split (must have image + depth files).')
parser.add_argument('--src', type=str, help='Path to read npy data files from.',
                    default='/home/'+os.environ['USER']+'/Dropbox/core_data/facies/train_data/')
parser.add_argument('--dst', type=str, help='Path to write jpeg files to.',
                    default='/home/'+os.environ['USER']+'/Dropbox/core_data/facies/label/')
parser.add_argument('--with_depth', dest='with_depth', action='store_true',
                    help='Flag to concurrently split+save depth arrays.')


def split_npy_image(well, src_path, dst_path, with_depth=False, max_rows=65000):
    """
    """
    img_arr = np.load(src_path / (well + '_image.npy'))
    depth_arr = np.load(src_path / (well + '_depth.npy'))

    assert img_arr.shape[0] == depth_arr.size, 'Image and depths must have same number of rows'

    height = depth_arr.size
    n_imgs = height // max_rows + 1

    tops = [i*max_rows for i in range(n_imgs)]
    bottoms = tops[1:] + [height]

    save_dir = dst_path / pathlib.Path(well)
    if not save_dir.exists():
        save_dir.mkdir()

    for t, b in zip(tops, bottoms):
        top = '{:.1f}'.format(depth_arr[t])
        bot = '_{:.1f}'.format(depth_arr[b-1])

        img_fname = save_dir / (top + bot + '.jpeg')
        print(f'Saving... {str(img_fname)}')
        io.imsave(str(img_fname), img_arr[t:b], quality=100)

        if with_depth:
            depth_fname = save_dir / (top + bot + '_depth.npy')
            print(f'Saving... {str(depth_fname)}')
            np.save(depth_fname, depth_arr[t:b])


if __name__ == '__main__':

    args = parser.parse_args()

    src_path = pathlib.Path(args.src)
    print(f'src: {str(src_path)}')
    dst_path = pathlib.Path(args.dst)
    print(f'dst: {str(dst_path)}')

    assert src_path.is_dir() and src_path.exists(), 'Check src_path'
    assert dst_path.is_dir() and dst_path.exists(), 'Check dst_path'

    split_npy_image(args.well, src_path, dst_path, with_depth=args.with_depth)
