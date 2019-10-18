# CoreBreakout

## Overview

`corebreakout` is a Python package built around [matterport/Mask\_RCNN](https://github.com/matterport/Mask_RCNN) for the segmentation and depth alignment of geological core images. It also provides the `CoreColumn` data structure for the saving, loading, and visualization of core image data.


## Target Platform

This package was developed and tested under Linux (Ubuntu, PopOS). It may work on other platforms, but probably requires adjustment of some configuration file parameters found at [corebreakout/defaults.py] (e.g., file system conventions for Windows).


## Requirements

The following Python packages are required:

- `numpy<=1.16.4`
- `matplotlib`
- `scikit-image`
- `tensorflow<=1.15`
- `mrcnn` via [matterport/Mask\_RCNN](https://github.com/matterport/Mask_RCNN)

The installation should check for and install `numpy`, `matplotlib`, and `scikit-image`.

The user is responsible for installing `tensorflow`, due to the ambiguity between `tensorflow` and `tensorflow-gpu`. We highly recommend the latter for building new models, although it may be possible to perform inference on CPU.

[Instructions for `Mask_RCNN` here]


## Installation

**Download:**

```
$ git clone https://github.com/rgmyr/corebreakout.git
$ cd corebreakout
```

Then install the package using **`pip`**. Develop mode installation (`-e`) is recommended, since many users will want to change some parameters to suit their particular dataset without having to reinstall afterward:
```
$ pip install -e .
```

## Data Assets

In order to use the provided dataset or perform inference without training a new model, you will need to download the data.

## Getting started

# Using new datasets

Third party tools are necessary for labeling new training images. There is built-in support for the default polygonal JSON annotation format of the [wkentaro/labelme](https://github.com/wkentaro/labelme) graphical image annotation tool, but any instance segmentation annotation format would be workable if the user is willing to write their own subclass of `mrcnn.utils.Dataset`.

For more details about `Dataset`s, see: [docs/creating_datasets.md]

# Development and Community Guidelines

## Submit an Issue

## Contributing

### Testing
