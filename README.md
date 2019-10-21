# CoreBreakout

### Overview

`corebreakout` is a Python package built around [matterport/Mask\_RCNN](https://github.com/matterport/Mask_RCNN) for the segmentation and depth-alignment of geological core sample images.

![](docs/images/JOSS_figure_workflow.png)

It also provides the `CoreColumn` data structure for saving, loading, manipulating, and visualizing depth-aligned core image data.

### Target Platform

This package was developed on Linux (Ubuntu, PopOS), and has also (TBD!) been tested on Mac OS X. It may work on other platforms, but we can make no guarantees.

## Installation

### Requirements

In addition to Python`>=3.6`, the following packages are required:

- `numpy<=1.16.4`
- `matplotlib`
- `scikit-image`
- `1.3<=tensorflow<=1.15`
- `mrcnn` via [matterport/Mask\_RCNN](https://github.com/matterport/Mask_RCNN)

The installation process should check for and install `numpy`, `matplotlib`, and `scikit-image`.

The `tensorflow` requirement is not explicitly enforced in `setup.py`, due to the ambiguity between `tensorflow` and `tensorflow-gpu` in versions `<=1.14`. We highly recommend the latter for building new models, although it may be possible to perform inference on CPU.

[Instructions for `Mask_RCNN` here]

### Download code

```
$ git clone --recurse-submodules https://github.com/rgmyr/corebreakout.git
$ cd corebreakout
```

### Download data (optional)

To make use of the provided dataset and model, or to train new a model starting from the pretrained COCO weights, you will need to download the `assets.zip` folder from the [releases page].

Unzip and place this folder in the root directory of the repository. If you would like to place it elsewhere, modify the paths in [corebreakout/defaults.py](https://github.com/rgmyr/corebreakout/blob/master/corebreakout/defaults.py) to point to your preferred location.


### Install (`conda` version)

We recommend installing `corebreakout` and its dependencies in an isolated environment.

If you are a `conda` user on a `*nix` system, you can create a new environment called `corebreakout` and install everything at once by running the provided `bash` script:

```
$ ./conda_install.sh
```

### Install (`pip` version)

First install `mrcnn` and its requirements:
```
$ cd Mask_RCNN
$ pip install -r requirements.txt
$ python setup.py install
```

Then install `corebreakout` using **`pip`**. Develop mode installation (`-e`) is recommended, since many users will want to change some parameters to suit their own data without having to reinstall afterward:
```
$ cd ..
$ pip install -e .
```

## Usage

### Using new datasets

Third party tools are necessary for labeling new training images. There is built-in support for the default polygonal JSON annotation format of the [wkentaro/labelme](https://github.com/wkentaro/labelme) graphical image annotation tool, but any instance segmentation annotation format would be workable if the user is willing to write their own subclass of `mrcnn.utils.Dataset`.

For details about `Dataset` usage and subclassing, see: [docs/creating_datasets.md](https://github.com/rgmyr/corebreakout/blob/master/docs/creating_datasets.md)

### Training models

Training a model requires a `Dataset`. You may (modify if necessary and) use [scripts/train_mrcnn_model.py](https://github.com/rgmyr/corebreakout/blob/master/scripts/train_mrcnn_model.py), or [notebooks/train_mrcnn_model.ipynb]().

For details about `mrcnn` model configuration and training, see: [docs/model_building.md](https://github.com/rgmyr/corebreakout/blob/master/docs/model_building.md)

### Processing images

Trained models can be used to instantiate and use a `CoreSegmenter` instance, with its `segment` and `segment_all` methods.

For details about image layout specification, see: [docs/layout_parameters.md](https://github.com/rgmyr/corebreakout/blob/master/docs/layout_parameters.md)

### Extracting depth ranges with OCR

We provide a script for extracting `top` and `base` depths from image text using `pytesseract`. This can help with aggregating the information required to process a large number of images.

You can install `pytesseract` via `conda` or `pip`, and then follow the instructions in the docstring of [scripts/get_ocr_depths.py](https://github.com/rgmyr/corebreakout/blob/master/scripts/train_mrcnn_model.py)


## Development and Community Guidelines

### Submit an Issue

- Navigate to the repository's [issue tab](https://github.com/rgmyr/corebreakout/issues)
- Search for existing related issues
- If necessary, create and submit a new issue

### Contributing

- Please see [`CONTRIBUTING.md`](.github/CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md) for how to contribute to the project

### Testing

- Most `corebreakout` functionality not requiring trained model weights can be verified with `pytest`:

```
$ cd <root_directory>
$ pytest .
```

- Model usage via the `CoreSegmenter` class can be verified by running `notebooks/test_inference.ipynb`
