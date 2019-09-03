---
title: 'Corebreakout: Raw Core Images to Structured Datasets'
tags:
  - Python
  - geology
  - geoscience
  - image processing
authors:
  - name: Ross G. Meyer
    orcid: 0000-0003-0872-7098
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Author 2
    orcid: 0000-0000-0000-0000
    affiliation: 2
affiliations:
 - name: Chevron Center of Research Excellence, Colorado School of Mines
   index: 1
 - name: Institution 2
   index: 2
date: 31 August 2019
bibliography: paper.bib
---

# Summary

In the field of geology, slabbed core samples are routinely used to study (answer?)
a variety of questions related to geologic history and processes. Unlike other
common sources of data, core preserves true subsurface heterogeneity and conditions
with no (minimal?) loss of resolution. However, logging core is a time consuming,
subjective, and still largely analog process. The digitization and structuring of
core image data allows for the development of automated and semi-automated workflows,
which can in turn facilitate large scale analysis of the hundreds of thousands of
meters of core stored in public and private repositories around the world.          

``corebreakout`` is a Python package for transforming raw images of geological core
sample material into structured data for analysis and modeling. It uses the Mask R-CNN
algorithm [@He:2017], and is built around the TensorFlow and Keras implementation
released by Matterport, Inc. [@Abdulla:2017]. We provide a labeled example dataset
courtesy of the British Geological Survey [citation, specific name], and make
it simple for geologists to add their own training images, build new models, and
subsequently process their own image datasets. It supports standard image layouts,
and provides several options for measuring and assigning depths to core sample
columns, including by labeling arbitrary "measuring stick" objects, or by specifying
hard-coded column endpoint coordinates.

``corebreakout`` is currently in use for ongoing work in image-based lithology
modeling, and is designed to be usable by geologists with a minimal background
in computation. The package is set up for standard command line and notebook use,
but may be extended to include a GUI in the future, which would enable a higher
degree of accuracy in depth assignment, and make the package usable even by those
with no programming experience.

# Figures

Include some example image processing figures: ![Example figure.](figure.png)

# Acknowledgements

We would like to acknowledge the contribution of open source image data from the BGS,
financial support from Chevron, etc...

# References
