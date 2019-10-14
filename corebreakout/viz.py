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
    display_instances(img, preds['rois'], preds['masks'], preds['class_ids'], class_names,
                     preds['scores'], colors=colors, ax=ax, figsize=figsize)


def draw_box(image, bbox, color, lw):
    """Draw RGB(A) `color` bounding box on image array."""
    y1, x1, y2, x2 = bbox
    image[y1:y1 + lw, x1:x2] = color
    image[y2:y2 + lw, x1:x2] = color
    image[y1:y2, x1:x1 + lw] = color
    image[y1:y2, x2:x2 + lw] = color
    return image


def draw_lines(img, coords, axis, color=[255,0,0], lw=10):
    """Draw `color` lines on `img` at `coords` along `axis`.

    axis == 0 --> horizonal lines
    axis == 1 --> vertical lines
    line width [`lw`] will round down to even numbers.

    NOTE: if any (coord +/- (lw // 2)) falls outside of `img`, will raise Exception.
    """
    assert axis in [0,1], '`axis` must be 0 (horizontal) or 1 (vertical)'

    hw = lw // 2
    if axis == 0:
        for row in coords:
            img[row-hw:row+hw+1,:,:] = color
    else:
        for col in coords:
            img[:,col-hw:col+hw+1,:] = color

    return img


def draw_box(img, box, color, lw):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    y1, x1, y2, x2 = box
    img[y1:y1 + lw, x1:x2] = color
    img[y2:y2 + lw, x1:x2] = color
    img[y1:y2, x1:x1 + lw] = color
    img[y1:y2, x2:x2 + lw] = color

    return img


###++++++++++++++++++++###
### Column depth ticks ###
###++++++++++++++++++++###
