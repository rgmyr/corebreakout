import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

ia.seed(1)

# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.9, 1.1), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-25, 25),
        shear=(-5, 5)
    )
], random_order=True) # apply augmenters in random order

def augment_images(X, y):
    '''Augument a batch of images X with associated segmentation masks y.'''
    seq_det = seq.to_deterministic()
    X_aug = seq_det.augment_images(X)
    y_aug = seq_det.augment_images(y)
    return X_aug, y_aug


def pad_batch_to_shape(X, shape):
    '''Pad a batch of images w/ shape (N, h, w, c) to (N, shape[0], shape[1], c).'''
    btch_pad = (0,0)
    h_pads = tuple([(shape[0] - X.shape[1])//2] * 2)
    w_pads = tuple([(shape[1] - X.shape[2])//2] * 2)
    chan_pad = (0,0)
    pads = ((0,0), h_pads, w_pads, (0,0))
    # TODO: add support for padding with some randomized color set
    return np.pad(x, pads, 'constant', constant_values=0.0)
