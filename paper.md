---
title: 'CoreBreakout: Subsurface Core Images to Depth-Registered Datasets'
tags:
  - Python
  - image processing
  - deep learning
  - geology
  - geoscience
  - subsurface
authors:
  - name: Ross G. Meyer
    orcid: 0000-0003-2344-554X
    affiliation: 1
  - name: Thomas Martin
    orcid: 0000-0002-4171-0004
    affiliation: 1
  - name: Zane R. Jobe
    orcid: 0000-0002-7654-4528
    affiliation: 1
affiliations:
 - name: Department of Geology and Geological Engineering, Colorado School of Mines
   index: 1
date: 9 December 2019
bibliography: paper.bib
---

# Summary

Core samples -- cylindrical rock samples taken from subsurface boreholes -- are commonly used by Earth scientists to study geologic history and processes. Core is usually cut into one-meter segments, slabbed lengthwise to expose a flat surface, and stored in cardboard or wooden boxes. Unlike other common sources of borehole data (*e.g.*, well logs), core is the only data that preserves true geologic scale and heterogeneity.

A geologist will often describe core by visual inspection and hand-draw a graphic log of the vertical changes in grain size and other rock properties (*e.g.*, @Jobe:2017). This description process is time consuming and subjective, and the resulting data is analog. The digitization and structuring of core image data allows for the development of automated and semi-automated workflows, which can in turn facilitate quantitative analysis of the millions of meters of core stored in public and private repositories around the world.

``corebreakout`` is a Python package for transforming raw images of geological core sample boxes into masked and depth-registered datasets for analysis and modeling. It uses the Mask R-CNN algorithm [@He:2017] for instance segmentation, and is built around the open source TensorFlow and Keras implementation released by Matterport, Inc. [@Abdulla:2017].


## Workflow

The primary user workflow enabled by ``corebreakout`` is depicted in Figure 1. We make it straightforward for geologists to add their own labeled training images using ``Labelme`` [@Wada:2016, @Russell:2007], configure and train new Mask R-CNN models on the labeled images, and subsequently use the trained models to process their own unlabeled images and compile depth-aligned datasets.

![Figure 1](JOSS_figure_workflow.png)


## Functionality

``corebreakout`` supports standard vertical and horizontal core image layouts, and provides several methods for measuring and assigning depths to core sample columns, including by labeling arbitrary "measuring stick" objects (*e.g.*, rulers, empty trays). We provide a labeled dataset courtesy of the British Geological Survey's OpenGeoscience project (https://www.bgs.ac.uk/data/bmd.html), as well as a Mask R-CNN model trained on this dataset for testing and demonstration.

In addition to the core Python package, the source code includes scripts for training models, extracting text meta-data from images with optical character recognition [@Smith:2007], and processing directories of images with saved models. The segmentation functionality could be integrated with a graphical user interface for an even higher degree of control and accuracy, but that is beyond the current scope of the project.

``corebreakout`` is currently being utilized for ongoing work in image-based lithology classification [@Martin:2019]. It has been used to compile a large image dataset for machine learning experiments, and the `CoreColumn` class provides a convenient interface for retrieving, manipulating, and visualizing stored data. We also plan to release our modeling code, which uses this data structure to combine depth-registered image data, sampled well log data, and interval labels into multi-modal datasets for sequence modeling.


# Acknowledgements

We would like to acknowledge the contribution of open source subsurface core images from the British Geological Survey (https://bgs.ac.uk/), and financial support from Chevron through the Chevron Center of Research Excellence at the Colorado School of Mines (https://core.mines.edu/).


# References
