# `corebreakout`

Package for segmentation, stacking, and depth alignment of geological core image datasets.

- Include a general pretrained model and tools for (re-)training with novel datasets.
- **other features**


## Target Platform

This package was developed and tested under Linux (Ubuntu). It may work on other platforms, but probably requires adjustment of some configuration file parameters (e.g., file system conventions for Windows).


## Requirements

The following packages are required with installation:

- numpy
- matplotlib
- scikit-image
- tensorflow
- matterport/Mask\_RCNN
- pycocotools

Other tools are necessary for labeling new training images. Any segmentation annotation format should be workable if the user is willing to write their own `Dataset` class. There is built-in support for the `COCO` format, which we generate using `labelme`.

**Post-labeling tools?**


## Installation

**Download:**

```
$ git clone https://github.com/rgmyr/corebreakout.git
```

**Install:**

```
$ cd corebreakout
```
And then use `pip`:
```
$ pip install -e .
```
**or** run setup:
```
$ python setup.py install
```

Develop mode installation (`pip install -e .`) is recommended, since users may want to change some parameters in the source code to suit their particular dataset of interest without having to reinstall, but it is not required.


## Tutorial

See the `tutorial` folder for notebooks demonstrating model training and usage.


### Additional object types for detection

One feature that could make this package more generally useful would be to have some options for using objects other than `core column` to determine the upper/lower pixel rows for cropping columns and generating accurate `depth` arrays.

I can see four basic options, each of which may work better or worse for any particular dataset:

- The hard-coded way we do it now. User specifies "layout(s)" that define the start/end of the core trays. This is a bit more of a hassle and probably sub-optimal, but it works well enough when the camera and tray positions are consistent.
- Taking the min/max of all the detected columns in an image. This would probably work well for times where you're pretty certain that at least one tray in each image will contain a full column worth of core.
- Detecting the core `box` as a seperate object, and using its bounding box to set the limits. This would work well for datasets with distinctive whole boxes (rather than e.g., partially occluded single trays), and where the core doesn't run over the edges of the boxes/trays too much.
- Detecting `scale` objects. This would similarly work well for datasets with distinctive scales that have relatively good constrast against the background (and that line up well with the actual core trays).

It wouldn't be too hard to include these as options, especially since `box` and `scale` would use pretty much (if not exactly) the same logic. Ultimately then it would be up to the user to decide which option is best for their dataset and whether they want to spend time labeling extra objects.


### Other TODO

- Support for `COCO` annotation format
    - Write a new `Dataset` class (utilize `pycocotools`)
    - Label some varied data from `pretrained/data/` (use `labelme` for polygon annotations)
    - Visualize some of the labels, make sure it comes out looking right

- Build `train` and `test` sets with new data, train some new models on it

- Move `CorePlotter` or parts of it into this package (check tick generation though -- make sure it works with `mode='collapse'`)

- Start on a write up for JOSS submission (?)
