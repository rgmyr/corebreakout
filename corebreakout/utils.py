"""
Image and segmentation utility functions.
"""
import numpy as np

# need to test out some orientation options
example_orientation = {
    'sort_axis' : 0,
    'sort_order' : +1,

}


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
    Convert boolean (H,W,N) array for N instances to integer (H,W) in range(0,N+1).
    """
    labels = np.zeros(masks.shape[0:-1])
    for i in range(masks.shape[-1]):
        labels += (i+1) * masks[:,:,i].astype(int)
    return labels.astype(int)


def region_crop(img, labels, region, endpts=endpts['A']):
    """
    Adjust region bbox and return cropped mask.

    TODO: make this handle both vertical and horizontal orientations.
    """
    x0, y0, x1, y1 = region.bbox
    if y0 > endpts[0]:
        y0 = endpts[0]
    if y1 < endpts[1]:
        y1 = endpts[1]
    region_img = img * np.expand_dims(labels==region.label, -1)
    return region_img[x0:x1,y0:y1,:]


def vstack_images(imgA, imgB):
    """
    Stack `imgA` and `imgB` arrays, after RHS zero-padding the narrower of the two, if necessary.
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
