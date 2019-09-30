"""
Utility functions. Mostly image + mask + region manipulation.
"""
import numpy as np

from mrcnn.visualize import display_instances


def show_preds(img, preds, class_names, figsize=(15,15)):
    """Less verbose wrapper for `mrcnn.visualize.display_instances`."""
    display_instances(img, preds['rois'], preds['masks'], preds['class_ids'],
                     class_names, preds['scores'], figsize=figsize)


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


def vstack_images(imgA, imgB):
    """
    Vstack `imgA` and `imgB` arrays, after RHS zero-padding the narrower if necessary.
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
