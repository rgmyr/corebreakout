"""Script utilizing `pytesseract` to extract top + base depths from image text with consistent location.

Instructions for users:
    - Modify `TEXT_BBOX` to match the location of informative text in your images
    - The BBOX should ideally be set such that the depths are the last two numbers in the text.

Notes:
    - Will walk <root_dir> and separately process any subdirectories containing <subdir> in their name.
    - Images in <subdir> should have '.jpeg' extension.
    - csv's with filenames, tops, and bottoms will be saved as <save_name>.csv in each <subdir> found
    - Whenever <2 float candidates are found in img[BBOX], 0.0 is used as the fill value
"""
import os
import re
import glob
import argparse

import pytesseract
from skimage import io
import pandas as pd
import matplotlib.pyplot as plt


# Set Tesseract arguments. See docs for options:
#    https://github.com/tesseract-ocr/tesseract/blob/master/doc/tesseract.1.asc#options
TESSERACT_CONFIG = '--psm 6'

# Set bounding box (x0, y0, x1, y1) of text
TEXT_BBOX = 200, 2200, 400, 2800


parser = argparse.ArgumentParser(description="Extract OCR top and base depths from images with `pytesseract`")
parser.add_argument('--root_dir',
    type=str,
    help="A common parent directory of all target <subdir> directories."
)
parser.add_argument('--subdir',
    type=str,
    default='converted',
    help="A string contained in the name of all target subdirectories."
)
parser.add_argument('--save_name',
    type=str,
    default='auto_depths',
    help="Name of depths csv file(s) to be saved in matching subdirs."
)
parser.add_argument('--force',
    dest='force',
    action='store_true',
    help="Flag to force overwrite of any existing <save_name>.csv files."
)
parser.add_argument('--inspect',
    dest='inspect',
    action='store_true',
    help="Flag to inspect images and print OCR output whenever there is an issue."
)


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
    """Take an image or string path to one, return (top, base) depths.

    If `inspect`, will `plt.show` any text bboxes where < 2 floats found.
    """
    x0, y0, x1, y1 = TEXT_BBOX
    crop_fn = lambda x: x[x0:x1,y0:y1]

    if isinstance(img, str):
        img = io.imread(img)

    # Get string of text, look for possible floats
    chars = pytesseract.image_to_string(crop_fn(img), config=TESSERACT_CONFIG)
    numbers = [truncate(number,2) for number in re.findall('\d+\.\d+', chars)]

    if len(numbers) < 2:
        if inspect:
            print('Less than two floats found, filling in 0\'s')
            print(chars)
            plt.imshow(crop_fn(img))
            plt.show()
        return (0.0, 0.0)
    else:
        # Return last two possible floats
        return (float(numbers[-2]), float(numbers[-1]))


def is_good_dir(d, subdir, save_name, force_overwrite):
    name_matches = (subdir in d.split('/')[-1])
    if force_overwrite:
        return name_matches
    else:
        no_depth_file = len(glob.glob(d+f'/*{save_name}.csv')) == 0
        return name_matches and no_depth_file


def find_subdirs(path, subdir, save_name, force_overwrite):
    subdirs = []
    for d in os.walk(path):
        if is_good_dir(d[0], subdir, save_name, force_overwrite):
            subdirs.append(d[0])
    return subdirs


def process_subdir(subdir_path, save_name, inspect):
    img_paths = list(sorted(glob.glob(subdir_path + '/*.jpeg')))
    img_files = [p.split('/')[-1] for p in img_paths]

    df = pd.DataFrame(index=img_files, columns=['top','bottom'])

    for img_path, img_file in zip(img_paths, img_files):
        top, bottom = depth_range_from_img(img_path, inspect)
        df.at[img_file,'top'] = top
        df.at[img_file,'bottom'] = bottom

    save_path = d+f'/{save_name}.csv'
    df.to_csv(save_path)
    print('Wrote file: ', save_path)


if __name__ == '__main__':

    args = parser.parse_args()

    img_dirs = find_subdirs(args.root_dir, args.subdir, args.save_name, args.force)

    print(f'Processing {len(img_dirs)} directories...')

    for d in img_dirs:
        process_subdir(d, args.save_name, args.inspect)
