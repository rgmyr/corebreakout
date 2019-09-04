---
title: 'CoreBreakout: Raw Subsurface Rock-Core Images to Structured, Depth-Registered Datasets'
tags:
  - Python
  - geology
  - geoscience
  - subsurface
  - image processing
authors:
  - name: Ross G. Meyer
    orcid: 0000-0003-2344-554X
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Thomas Martin
    orcid: 0000-0002-4171-0004
    affiliation: 1
  - name: Zane R. Jobe
    orcid: 0000-0002-7654-4528
    affiliation: 1
affiliations:
 - name: Department of Geology and Geological Engineering, Colorado School of Mines
   index: 1
date: 31 August 2019
bibliography: paper.bib
---

# Summary

Core samples, which are cylindrical rock samples taken from subsurface boreholes, are commonly used by Earth scientists to answer a variety of questions related to geologic history and processes. Unlike other common sources of borehole data (e.g., well logs), core is the only data that preserves true geologic scale and heterogeneity. Most often, a geologist will describe the core via visual inspection and hand-draw a graphic log of the lithologic heterogeneity (i.e., the vertical changes in grain size and other rock properties). This description process is time consuming, subjective, and analog. The digitization and structuring of core image data allows for the development of semi-automated workflows, which can in turn facilitate analysis of the hundreds of thousangs of meters of core stored in public and private repositories around the world.


``corebreakout`` is a Python package for transforming raw images of geological core samples into structured data for analysis and modeling. It uses the Mask R-CNN algorithm [@He:2017], and is built around the TensorFlow and Keras implementation released by Matterport, Inc. [@Abdulla:2017]. We provide a labeled example dataset courtesy of the British Geological Survey, and make it simple for geologists to add their own training images, build new models, and subsequently process their own image datasets. It supports all standard  image layouts, and provides several options for measuring and assigning depths to core sample columns, including by labeling arbitrary "measuring stick" objects, or by specifying hard-coded column endpoint coordinates.

``corebreakout`` is currently being utilized for ongoing work in image-based lithology modeling (Martin et al., 2019), and is designed to be usable by geologists with a minimal background in computation. The package is set up for command line and notebook use, but may be extended to include a GUI in the future, which would enable a higher degree of accuracy in depth assignment, and make the package usable even by those with no programming experience.

# Figures

Include some example image processing figures: ![Example figure.](figure.png)

# Acknowledgements

We would like to acknowledge the contribution of open source subsurface core images from the British Geological Survey (https://bgs.ac.uk/), and financial support from Chevron through the Chevron Center of Research Excellence (https://core.mines.edu/).

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# References
