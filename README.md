# CoreBreakout

[![status](https://joss.theoj.org/papers/add2021f95268fd4cd2850b105f3d570/status.svg)](https://joss.theoj.org/papers/add2021f95268fd4cd2850b105f3d570)

Requirements, installation, and contribution guidelines can be found below. Our full usage and API documentation can be found at: [corebreakout.readthedocs.io](https://corebreakout.readthedocs.io/en/latest/)

### Overview

`corebreakout` is a Python package built around [matterport/Mask\_RCNN](https://github.com/matterport/Mask_RCNN) for the segmentation and depth-alignment of geological core sample images. It provides utilities and an API to enable the workflow depicted in the figure below, as well as a `CoreColumn` data structure to manage and manipulate the resulting depth-registered image data:

![](JOSS_figure_workflow.png)

We are currently using this package to enable research on [Lithology Prediction of Slabbed Core Photos Using Machine Learning Models](https://figshare.com/articles/Lithology_Prediction_of_Slabbed_Core_Photos_Using_Machine_Learning_Models/8023835/2), and are working on getting a DOI for the project through the [Journal of Open Source Software](https://joss.theoj.org/).

## Getting Started

### Target Platform

This package was developed on Linux (Ubuntu, PopOS), and has also been tested on OS X. It may work on other platforms, but we make no guarantees.

### Requirements

In addition to Python`>=3.6`, the packages listed in [requirements.txt](requirements.txt) are required. Notable exceptions to the list are:

- `1.3<=tensorflow-gpu<=1.14` (or possibly just `tensorflow`)
- `mrcnn` via [submodule: matterport/Mask\_RCNN](https://github.com/matterport/Mask_RCNN/tree/3deaec5d902d16e1daf56b62d5971d428dc920bc)

The TensorFlow requirement is not explicitly listed in `requirements.txt` due to the ambiguity between `tensorflow` and `tensorflow-gpu` in versions `<=1.14`. The latter is almost certainly required for training new models, although it may be possible to perform inference with saved models on CPU, and use of the `CoreColumn` data structure does not require a GPU.

Note that TensorFlow GPU capabilities are implemented with [CUDA](https://developer.nvidia.com/cuda-zone), which requires a [supported NVIDIA GPU](https://developer.nvidia.com/cuda-gpus).

#### Additional (Optional) Requirements

Optionally, `jupyter` is required to run demo and test notebooks, and `pytest` is required to run unit tests. Both of these should be manually installed if you plan to modify or contribute to the package source code.

We also provide a script for extraction of top/base depths from core image text using `pytesseract`. After installing the [Tesseract OCR Engine](https://github.com/tesseract-ocr/tesseract) on your machine, you can install the `pytesseract` package with `conda` or `pip`.

### Download code

```
$ git clone --recurse-submodules https://github.com/rgmyr/corebreakout.git
$ cd corebreakout
```

### Download data (optional)

To make use of the provided dataset and model, or to train new a model starting from the pretrained COCO weights, you will need to download the `assets.zip` folder from the [v0.2 Release](https://github.com/rgmyr/corebreakout/releases/tag/v0.2).

Unzip and place this folder in the root directory of the repository (its contents will be ignored by `git` -- see the `.gitignore`). If you would like to place it elsewhere, you should modify the paths in [corebreakout/defaults.py](https://github.com/rgmyr/corebreakout/blob/master/corebreakout/defaults.py) to point to your preferred location.

The current version of `assets/data` has JSON annotation files which include an `imageData` field representing the associated images as strings. For now you can delete this field and reduce the size of the data with `scripts/prune_imageData.py`:

```
$ python scripts/prune_imageData.py assets/
```

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

Please refer to our [readthedocs page](https://corebreakout.readthedocs.io/en/latest/) for full documentation!

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
