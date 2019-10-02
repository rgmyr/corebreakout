"""
Script utilizing `pytesseract` to extract top + base depths from image text.
"""
import os
import re
import glob
import argparse

import pytesseract
from skimage import io
import pandas as pd
import matplotlib.pyplot as plt

# Set Tesseract arguments
# Docs: https://github.com/tesseract-ocr/tesseract/blob/master/doc/tesseract.1.asc#options
TESSERACT_CONFIG = '-psm 6'

TEXT_BBOX = 200, 2200, 400, 2800


def truncate(f, n):
    """Truncate or pad a float `f` to `n` decimal places, without rounding."""
    if isinstance(f, float):
        s = '{}'.format(f)
    else:
        s = f
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


def depth_range_from_img(img, inspect):
    # defaults for now, seems to work for both layouts in schiehallion
    crop_fn = lambda x: x[200:400,2200:2800]

    if isinstance(img, str):
        img = io.imread(img)

    chars = pytesseract.image_to_string(crop_fn(img), config=TESSERACT_CONFIG)
    numbers = [truncate(number,2) for number in re.findall('\d+\.\d+', chars)]

    if len(numbers) < 2:
        if inspect:
            print('PROBLEM WITH THIS FILE, filling in 0\'s')
            print(chars)
            plt.imshow(cropper(img))
            plt.show()
        return (0.0, 0.0)
    else:
        return (float(numbers[-2]), float(numbers[-1]))


def is_good_dir(d, force_overwrite):
    is_converted_dir = d.split('/')[-1] == 'converted'
    if force_overwrite:
        return is_converted_dir
    else:
        no_depth_file = len(glob.glob(d+'/*auto_depths.csv')) == 0
        return is_converted_dir and no_depth_file


def auto_label_all_subdirs(path, force_overwrite=False, inspect=False):
    img_dirs = [d[0] for d in os.walk(path) if is_good_dir(d[0], force_overwrite)]
    for d in img_dirs:
        print('Processing directory: ', d)
        img_paths = list(sorted(glob.glob(d+'/*.jpeg')))
        img_files = [p.split('/')[-1] for p in img_paths]
        df = pd.DataFrame(index=img_files, columns=['top','bottom'])
        for img_path, img_file in zip(img_paths, img_files):
            top, bottom = depth_range_from_img(img_path, inspect)
            df.at[img_file,'top'] = top
            df.at[img_file,'bottom'] = bottom
        df.to_csv(d+'/auto_depths.csv')
        print('Wrote file: .../auto_depths.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'root_dir',
        type=str,
        help="A common parent directory of all target `converted` directories."
    )
    parser.add_argument(
        '--force',
        dest='force',
        action='store_true',
        help="Flag to force overwrite of any existing auto_depths.csv files"
    )
    parser.add_argument(
        '--inspect',
        dest='inspect',
        action='store_true',
        help="Flag to inspect images and OCR output whenever there is an issue"
    )
    args = parser.parse_args()
    auto_label_all_subdirs(args.root_dir, force_overwrite=args.force, inspect=args.inspect)
