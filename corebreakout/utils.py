"""
Some random utility functions.
"""
import numpy as np


def vstack_images(imgA, imgB)
    """
    Stack `imgA` and `imgB`; after RHS zero-padding the narrower, if necessary.
    """
    dw = imgA.shape[1] - imgB.shape[1]

    if dw == 0:
        return np.concatenate([self.img, other.img])
    else:
        pads = ((0,0), (0, abs(dw)), (0,0))
    
    if dw < 0:
        paddedA = np.pad(imgA, pads, 'constant')
        return np.concatenate([paddedA, imgB])
    else:
        paddedB = np.pad(imgB, pads, 'constant')
        return np.concatenate([imgA, paddedB])


