"""
Image / mask / region manipulation, visualization.
"""
import numpy as np

from mrcnn.visualize import display_instances


def v_overlapping(r0, r1):
    """Check if skimage regions `r0` & `r1` are *vertically* overlapping."""
    return (r0.bbox[0] < r1.bbox[2] and r0.bbox[2] > r1.bbox[0])


def h_overlapping(r0, r1):
    """Check if skimage regions `r0` & `r1` are *horizontally* overlapping."""
    return (r0.bbox[1] < r1.bbox[3] and r0.bbox[3] > r1.bbox[1])


def masks_to_labels(masks):
    """Convert boolean (H,W,N) `masks` array to integer (H,W) in range(0,N+1)."""
    labels = np.zeros(masks.shape[0:-1])
    for i in range(masks.shape[-1]):
        labels += (i+1) * masks[:,:,i].astype(int)
    return labels.astype(int)


def squeeze_labels(labels):
    """Set labels to range(0, objects+1)"""
    label_ids = np.unique([r.label for r in measure.regionprops(labels)])

    for new_label, label_id in zip(range(1, label_ids.size), label_ids[1:]):
        labels[labels==label_id] == new_label

    return labels


def vstack_images(imgA, imgB):
    """Vstack `imgA` and `imgB`, after RHS zero-padding the narrower if necessary."""
    dimA, dimB = imgA.ndim, imgB.ndim
    assert dimA == dimB, f'Cannot vstack images of different dimensions: {(dimA, dimB)}'
    assert dimA in [2, 3], f'Images must be 2D or 3D, not {dimA}D'

    dw = imgA.shape[1] - imgB.shape[1]

    if dw == 0:
        return np.concatenate([imgA, imgB])
    elif dimA == 2:
        pads = ((0,0), (0, abs(dw)))
    else:
        pads = ((0,0), (0, abs(dw)), (0,0))

    if dw < 0:
        paddedA = np.pad(imgA, pads, 'constant')
        return np.concatenate([paddedA, imgB])
    else:
        paddedB = np.pad(imgB, pads, 'constant')
        return np.concatenate([imgA, paddedB])


###+++++++++++++++###
### Visualization ###
###+++++++++++++++###

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
