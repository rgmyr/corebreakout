import glob
import numpy as np
from skimage import io, measure, morphology
from itertools import combinations


def morphology_op(remove_holes=False, closing_selem=None, erosion_selem=None):
    """Create a function chaining multiple binary morphology operations.

    Parameters
    ----------
    holes : boolean, optional
        Specify True to call skimage.morphology.remove_small_holes
    closing_selem : np.array selection element
        Selection element from skimage.morphology to use for binary closing
    morphology_func : np.array selection element
        Selection element from skimage.morphology to use for binary erosion

    Returns
    -------
    op : function
        Chain of specified morphological operations, taking and returning a binary mask
    """

    def _op(mask):
        if remove_holes:
            mask = morphology.remove_small_holes(mask)
        if closing_selem:
            mask = morphology.binary_closing(mask, selem=closing_selem)
        if erosion_selem:
            mask = morphology.binary_erosion(mask, selem=erosion_selem)
        return mask

    return _op


def generate_mask(img, masking_color, tolerance=0, morph_op=None):
    """Generate a mask from a manually painted RGB image.

    Parameters
    ----------
    img : np.array, ndim=3
        RGB image as numpy array.
    masking_color : tuple, len=3
        (R, G, B) tuple to treat as TRUE pixel value
    tolerance : int, optional
        Allowable deviance from `masking_color` (per channel)
    morph_op : function (optional)
        Function to be applied to binary mask before returning
    Returns
    -------
    mask : np.array, ndim=2
        Binary mask, equivalent to where img == color
    """

    R, G, B = masking_color
    reds, greens, blues = tuple(img[:, :, i] for i in range(3))
    mask = np.logical_and.reduce(
        (
            np.abs(reds - R) <= tolerance,
            np.abs(greens - G) <= tolerance,
            np.abs(blues - B) <= tolerance,
        )
    )
    if morph_op:
        mask = morph_op(mask)

    return mask.astype(np.uint8)


def clean_blue(image):
    image = morphology.remove_small_holes(image, area_threshold=500)
    image = morphology.remove_small_objects(image, min_size=250)
    return image


def make_labels(blue_img, masking_color=(2, 0, 251), show=False):
    """Should call this collapse_labels or something?"""
    mask = measure.label(
        generate_mask(blue_img, masking_color, morph_op=clean_blue), background=0
    )
    init_regions = measure.regionprops(mask)

    for r0, r1 in combinations(init_regions, 2):
        if overlapping(r0, r1):
            mask[mask == r1.label] = r0.label

    # TABLET: draw_box(box_img, [30,1700,800,2850], (0,255,0), 30)
    if show:
        box_img = cimg
        for r in measure.regionprops(mask):
            box_img = draw_box(box_img, r.bbox, (255, 0, 0), 30)
        plt.figure(figsize=(25, 25))
        plt.imshow(box_img)
        plt.imshow(mask, cmap="jet", alpha=0.2)
        plt.show()

    return mask


'''
def squeeze_labels(mask):
    """set labels to range(0, objects+1)"""
    labels = np.unique([r.label for r in measure.regionprops(mask)])

    for new_label, label in zip(range(1, labels.size), labels[1:]):
        mask[mask==label] == new_label

    return mask

for cimg, limg in zip(core_imgs, label_imgs):
'''
