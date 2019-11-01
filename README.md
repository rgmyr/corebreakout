# CoreBreakout

### Overview

`corebreakout` is a Python package built around [matterport/Mask\_RCNN](https://github.com/matterport/Mask_RCNN) for the segmentation and depth-alignment of geological core sample images.

![](JOSS_figure_workflow.png)

It also provides a `CoreColumn` data structure, which can be sliced and added, and which facilitates saving, loading, manipulating, and visualizing depth-aligned core image data.

We are currently using this package to enable research on [Lithology Prediction of Slabbed Core Photos Using Machine Learning Models](https://figshare.com/articles/Lithology_Prediction_of_Slabbed_Core_Photos_Using_Machine_Learning_Models/8023835/2), and are working on getting a DOI for the project through the [Journal of Open Source Software](https://joss.theoj.org/).

## Getting Started

### Target Platform

This package was developed on Linux (Ubuntu, PopOS), and has also been tested on OS X. It may work on other platforms, but we can make no guarantees.

### Requirements

In addition to Python`>=3.6`, the packages listed in [requirements.txt](requirements.txt) are required. Notable exceptions to the list are:

- `1.3<=tensorflow-gpu<=1.14` (or possibly just `tensorflow`)
- `mrcnn` via [submodule: matterport/Mask\_RCNN](https://github.com/matterport/Mask_RCNN/tree/3deaec5d902d16e1daf56b62d5971d428dc920bc)

Optionally, `jupyter` is required to run demo and test notebooks, and `pytest` is required to run unit tests.

The TensorFlow requirement is not explicitly listed in `requirements.txt` due to the ambiguity between `tensorflow` and `tensorflow-gpu` in versions `<=1.14`. The latter is almost certainly required for building new models, although it may be possible to perform inference with saved models on CPU.

Note that TensorFlow GPU capabilities are implemented with [CUDA](https://developer.nvidia.com/cuda-zone), which requires a [supported NVIDIA GPU](https://developer.nvidia.com/cuda-gpus).

### Download code

```
$ git clone --recurse-submodules https://github.com/rgmyr/corebreakout.git
$ cd corebreakout
```

### Download data (optional)

To make use of the provided dataset and model, or to train new a model starting from the pretrained COCO weights, you will need to download the `assets.zip` folder from the [releases page].

Unzip and place this folder in the root directory of the repository. If you would like to place it elsewhere, you should modify the paths in [corebreakout/defaults.py](https://github.com/rgmyr/corebreakout/blob/master/corebreakout/defaults.py) to point to your preferred location.


### Installation

We recommend installing `corebreakout` and its dependencies in an isolated environment, and further recommend the use of `conda`. See [Conda: Managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

---

To create a new `conda` environment called `corebreakout-env` and activate it:

```
$ conda create -n corebreakout-env python=3.6 tensorflow-gpu=1.14
$ conda activate corebreakout-env
```

**Note:** If you want to try a CPU-only installation, then replace `tensorflow-gpu` with `tensorflow`. You may also lower the version number if you are on a machine with `CUDA<10.0` (required for TensorFlow`>=1.13`). See [TensorFlow GPU requirements](https://www.tensorflow.org/install/gpu#software_requirements) for more compatibility details.

---

Then install the rest of the required packages into the environment:

```
$ conda install --file requirements.txt
```

---

Finally, install `mrcnn` and `corebreakout` using `pip`. Develop mode installation (`-e`) is recommended (but not required) for `corebreakout`, since many users will want to change some of the default parameters to suit their own data without having to reinstall afterward:

```
$ pip install ./Mask_RCNN
$ pip install -e .
```

## Usage

### Using new datasets

Third party tools are necessary for labeling new training images. There is built-in support for the default polygonal JSON annotation format of the [wkentaro/labelme](https://github.com/wkentaro/labelme) graphical image annotation tool, but any instance segmentation annotation format would be workable if the user is willing to write their own subclass of `mrcnn.utils.Dataset`.

For details about `Dataset` usage and subclassing, see: [docs/creating_datasets.md](https://github.com/rgmyr/corebreakout/blob/master/docs/creating_datasets.md)

### Training models

Training a model requires a `Dataset`. You may run [scripts/train_mrcnn_model.py](https://github.com/rgmyr/corebreakout/blob/master/scripts/train_mrcnn_model.py) (after modifying, if necessary), or [notebooks/train_mrcnn_model.ipynb]().

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

- Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md) for how to contribute to the project

### Testing

- Most `corebreakout` functionality not requiring trained model weights can be verified with `pytest`:

```
$ cd <root_directory>
$ pytest .
```

- Model usage via the `CoreSegmenter` class can be verified by running `tests/notebooks/test_inference.ipynb` (requires saved model weights)
- Plotting of `CoreColumn`s can be verified by running `tests/notebooks/test_plotting.ipynb`
