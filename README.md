# `corebreakout`

Python package built around [matterport/Mask\_RCNN](https://github.com/matterport/Mask_RCNN) for segmentation, stacking, and depth alignment of geological core image datasets.

### TODO

- Move `CorePlotter` or parts of it into this package (check tick generation though -- make sure it works with `mode='collapse'`)

- Write tutorials: one for building/training models, one for performing inference and processing datasets

- Clean up `scripts/` and `notebooks` directories

- Finishing writing `pytest` files


## Target Platform

This package was developed and tested under Linux (Ubuntu, PopOS). It may work on other platforms, but probably requires adjustment of some configuration file parameters found at [corebreakout/defaults.py] (e.g., file system conventions for Windows).


## Requirements

The following Python packages are required:

- `numpy`
- `matplotlib`
- `scikit-image`
- `tensorflow`
- `mrcnn` via [matterport/Mask\_RCNN](https://github.com/matterport/Mask_RCNN)

The installation should check for and install `numpy`, `matplotlib`, and `scikit-image`. The user should install others...

Using `tensorflow-gpu` is very much recommended.

# Using new datasets

Third party tools are necessary for labeling new training images. There is built-in support for the default polygonal JSON annotation format of the [wkentaro/labelme](https://github.com/wkentaro/labelme) graphical image annotation tool, but any instance segmentation annotation format would be workable if the user is willing to write their own subclass of `mrcnn.utils.Dataset`.

See more detailed documentation: [docs/creating_datasets.md]

**Post-labeling tools?**


## Installation

**Download:**

```
$ git clone https://github.com/rgmyr/corebreakout.git
$ cd corebreakout
```

And then use **`pip`**:
```
$ pip install -e .
```
**OR** run `setup.py`:
```
$ python setup.py install
```

Develop mode installation (`pip install -e .`) is recommended, since we expect that most users will want to change some parameters in the source code to suit their particular dataset without having to reinstall afterward.


## Tutorial

See the `docs/tutorial` folder for notebooks demonstrating model training and usage.


### Additional object types for detection

One feature that could make this package more generally useful would be to have some options for using objects other than `core column` to determine the upper/lower pixel rows for cropping columns and generating accurate `depth` arrays.

I can see four basic options, each of which may work better or worse for any particular dataset:

- The hard-coded way we do it now. User specifies "layout(s)" that define the start/end of the core trays. This is a bit more of a hassle and probably sub-optimal, but it works well enough when the camera and tray positions are consistent.
- Taking the min/max of all the detected columns in an image. This would probably work well for times where you're pretty certain that at least one tray in each image will contain a full column worth of core.
- Detecting the core `box` as a seperate object, and using its bounding box to set the limits. This would work well for datasets with distinctive whole boxes (rather than e.g., partially occluded single trays), and where the core doesn't run over the edges of the boxes/trays too much.
- Detecting `scale` objects. This would similarly work well for datasets with distinctive scales that have relatively good constrast against the background (and that line up well with the actual core trays).

It wouldn't be too hard to include these as options, especially since `box` and `scale` would use pretty much (if not exactly) the same logic. Ultimately then it would be up to the user to decide which option is best for their dataset and whether they want to spend time labeling extra objects.


### Other TODO

- Label some varied data from `pretrained/data/` (use `labelme` for polygon annotations)

- Build `train` and `test` sets with new data, train some new models on it

- Move `CorePlotter` or parts of it into this package (check tick generation though -- make sure it works with `mode='collapse'`)
