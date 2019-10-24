"""
Assorted visualization functions.
"""
from mrcnn.visualize import display_instances


###++++++++++++++++++++++###
### Model + bbox + lines ###
###++++++++++++++++++++++###


def show_preds(img, preds, class_names, colors=None, ax=None, figsize=(16, 16)):
    """Less verbose wrapper for `mrcnn.visualize.display_instances`.

    Parameters
    ----------
    colors : list or array, optional
        Colors to use for each object in `preds`, default is random color for each.
    ax : matplotlib axis, optional
        An axis to plot onto. If None, will create one with size `figsize`.
    """
    display_instances(
        img,
        preds["rois"],
        preds["masks"],
        preds["class_ids"],
        class_names,
        preds["scores"],
        colors=colors,
        ax=ax,
        figsize=figsize,
    )


def draw_box(image, bbox, color, lw):
    """Draw RGB(A) `color` bounding box on image array."""
    y1, x1, y2, x2 = bbox
    image[y1 : y1 + lw, x1:x2] = color
    image[y2 : y2 + lw, x1:x2] = color
    image[y1:y2, x1 : x1 + lw] = color
    image[y1:y2, x2 : x2 + lw] = color
    return image


def draw_lines(img, coords, axis, color=[255, 0, 0], lw=10):
    """Draw `color` lines on `img` at `coords` along `axis`.

    axis == 0 --> horizonal lines
    axis == 1 --> vertical lines
    line width [`lw`] will round down to even numbers.

    NOTE: if any (coord +/- (lw // 2)) falls outside of `img`, will raise Exception.
    """
    assert axis in [0, 1], "`axis` must be 0 (horizontal) or 1 (vertical)"

    hw = lw // 2
    if axis == 0:
        for row in coords:
            img[row - hw : row + hw + 1, :, :] = color
    else:
        for col in coords:
            img[:, col - hw : col + hw + 1, :] = color

    return img


def draw_box(img, box, color, lw):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    y1, x1, y2, x2 = box
    img[y1 : y1 + lw, x1:x2] = color
    img[y2 : y2 + lw, x1:x2] = color
    img[y1:y2, x1 : x1 + lw] = color
    img[y1:y2, x2 : x2 + lw] = color

    return img


###++++++++++++++++++++###
### Column depth ticks ###
###++++++++++++++++++++###


def make_depth_ticks(
    depths,
    major_precision=0.1,
    major_format_str="{:.1f}",
    minor_precision=0.01,
    minor_format_str="{:.2f}",
):
    """Generate major & minor (ticks, locs) for depth array axis.

    Parameters
    ----------
    depths : array
        An array of (ordered) depth values from which to generate ticks/locs.
    *_precision : float, optional
        Major, minor tick spacing (in depth units), defaults=0.1, 0.01.
    *_format_str : str, optional
        Format strings to coerce depths -> tick strings, defaults='{:.1f}', '{:.2f}'.

    Returns
    -------
    major_ticks, major_locs, minor_ticks, minor_locs

    *_ticks : lists of tick label strings
    *_locs : lists of tick locations in array coordinates (fractional indices)
    """
    # lambdas to convert values --> strs
    major_fmt_fn = lambda x: major_format_str.format(x)
    minor_fmt_fn = lambda x: minor_format_str.format(x)

    major_ticks, major_locs = [], []
    minor_ticks, minor_locs = [], []

    # remainders of depth w.r.t. precision
    major_rmndr = np.insert(self.depths % major_precision, (0, self.height), np.inf)
    minor_rmndr = np.insert(self.depths % minor_precision, (0, self.height), np.inf)

    for i in np.arange(1, self.height + 1):

        if np.argmin(major_rmndr[i - 1 : i + 2]) == 1:
            major_ticks.append(major_fmt_fn(self.depths[i - 1]))
            major_locs.append(i)

        elif np.argmin(minor_rmndr[i - 1 : i + 2]) == 1:
            # if already major tick, don't bother
            # NOTE: ugh, need to fix again
            if major_ticks[-1] == major_fmt_fn(self.depths[i - 1]):
                continue
            minor_ticks.append(minor_fmt_fn(self.depths[i - 1]))
            minor_locs.append(i)

    # get last tick if needed, doesn't work above for some reason
    last_depth = np.round(self.depths[-1], decimals=1)
    if (last_depth % 1.0) == 0.0:
        major_ticks.append(major_fmt_fn(last_depth))
        major_locs.append(self.height - 1)

    return major_ticks, major_locs, minor_ticks, minor_locs
