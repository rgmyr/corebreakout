"""
Utility functions. Mostly image + mask + region manipulation.
"""
import numpy as np

from mrcnn.visualize import display_instances


def show_preds(img, preds, dataset, figsize=(15,15)):
    """
    Less verbose wrapper for `mrcnn.visualize.display_instances`
    """
    display_instances(img, preds['rois'], preds['masks'], preds['class_ids'], dataset.class_names, preds['scores'], figsize=figsize)


def v_overlapping(r0, r1):
    """
    Check if skimage regions `r0` & `r1` are *vertically* overlapping.
    """
    return (r0.bbox[0] < r1.bbox[2] and r0.bbox[2] > r1.bbox[0])

def h_overlapping(r0, r1):
    """
    Check if skimage regions `r0` & `r1` are *horizontally* overlapping.
    """
    return (r0.bbox[1] < r1.bbox[3] and r0.bbox[3] > r1.bbox[1])


def masks_to_labels(masks):
    """
    Convert boolean (H,W,N) `masks` array for N instances to integer (H,W) in range(0,N+1).
    """
    labels = np.zeros(masks.shape[0:-1])
    for i in range(masks.shape[-1]):
        labels += (i+1) * masks[:,:,i].astype(int)
    return labels.astype(int)


def region_crop(img, labels, region, axis=0, endpts=(815, 6775)):
    """
    Adjust region bbox and return cropped region * mask.

    Parameters
    ----------
    img : array
        The image to crop
    labels : array
        Mask of integer labels, same height and width as `img`
    region : skimage.RegionProperties instance
        Region object corresponding to column to crop around
    axis : int, optional
        Which axis to change `endpts` along, default=0 (y-coordinates)
    endpts : tuple(int)
        Most extreme endpoint coordinates allowed along `axis`

    Returns
    -------
    region : array
        Masked image region, cropped in (adjusted) bounding box
    """
    r0, c0, r1, c1 = region.bbox

    if axis is 0:
        c0, c1 = min(c0, endpts[0]), max(c1, endpts[1])
    elif axis is 1:
        r0, r1 = min(r0, endpts[0]), max(r1, endpts[1])

    region_img = img * np.expand_dims(labels==region.label, -1)

    return region_img[r0:r1,c0:c1,:]


def vstack_images(imgA, imgB):
    """
    Stack `imgA` and `imgB` arrays vertically, after RHS zero-padding the narrower if necessary.
    """
    dimA, dimB = imgA.ndim, imgB.ndim
    assert dimA == dimB, f'Cannot vstack images of different dimensions: {(dimA, dimB)}'
    assert dimA in [2, 3], f'Images must be 2D or 3D, not {dimA}D'

    dw = imgA.shape[1] - imgB.shape[1]

    if dw == 0:
        return np.concatenate([self.img, other.img])
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
