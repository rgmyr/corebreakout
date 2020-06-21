.. CoreBreakout documentation master file, created by
   sphinx-quickstart on Fri Apr 17 06:21:33 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CoreBreakout's documentation!
========================================

Project Repository: `rgmyr/corebreakout <https://github.com/rgmyr/corebreakout>`_

``mrcnn`` Repository: `matterport/Mask_RCNN <https://github.com/matterport/Mask_RCNN>`_

``corebreakout`` provides two main functionalities: **(1)** a deep learning workflow for transforming raw images of geological core sample boxes into depth-registered datasets, which is facilitated by the ``CoreSegmenter`` API, and **(2)** a ``CoreColumn`` data structure for storing and manipulating depth-registered image data.

|workflowfigure|

This documentation covers usage of ``corebreakout``. To dig into the finer details of the Mask R-CNN model implementation, please refer to ``mrcnn``'s documentation.

Provided Data
=============

We provide labeled images from the British Geological Survey's North Sea collection, as well as a saved Mask R-CNN model trained on this data, and a pretrained COCO model from which to start new training runs.

To use the data or models, download the ``assets.zip`` folder from the `Releases Page <https://github.com/rgmyr/corebreakout/releases>`_. Unzip it in the root directory of the project, or modify the paths in `defaults.py <https://github.com/rgmyr/corebreakout/blob/master/corebreakout/defaults.py>`_ to point to the location.

JSON annotations in this data currently contain a superfluous field called ``imageData`` which takes up most of the memory for these files. You can delete this field and reduce the file sizes with ``scripts/prune_imageData.py``:

.. code::

  $ python scripts/prune_imageData.py assets/

If you want to use your own data, then label some images for Mask R-CNN training, following the guidelines in :ref:`creating-datasets`. We recommend starting with 20-30 images.

Overview
========

Image Processing Workflow
-------------------------

(1) If you're looking to use your own data, you will probably need to label some of your images for best results. Follow the guidelines in :ref:`creating-datasets`.

(2) Train a Mask R-CNN model using your labeled data. Model configuration, training, and selection are explained in :ref:`model-building`.

(3) Use the trained model to process directories of unlabeled images and save the results as a ``CoreColumn``. We provide `scripts/process_directory.py <https://github.com/rgmyr/corebreakout/blob/master/scripts/process_directory.py>`__ to facilitate this step. It requires saved model weights, and a csv file listing the top and bottom depths for each image in the directory. To make creating these csv's easier, we provide `scripts/get_ocr_depths.py <https://github.com/rgmyr/corebreakout/blob/master/scripts/get_ocr_depths.py>`_.

The ``CoreSegmenter`` Class
---------------------------

`scripts/process_directory.py <https://github.com/rgmyr/corebreakout/blob/master/scripts/process_directory.py>`__ uses the ``corebreakout.CoreSegmenter`` API to handle converting images to ``CoreColumns``, and you may also use this class directly:

.. code:: python

  segmenter = corebreakout.CoreSegmenter(
        model_dir,
        weights_path,
        model_config  = corebreakout.defaults.DefaultConfig,
        class_names   = corebreakout.defaults.CLASSES,
        layout_params = corebreakout.defaults.LAYOUT_PARAMS
  )

  # `img` can be an array or a path to an image
  column = segmenter.segment(img, [top_depth, base_depth], **kwargs)

  # for iterables of images (or paths) and depth range pairs
  column = segmenter.segment_all(imgs, depth_ranges, **kwargs)

``class_names`` should correspond to those in the dataset on which the model was trained, and ``layout_params`` are explained in detail in the :ref:`layout-parameters` documentation.

The ``CoreColumn`` Class
------------------------

This object is a container for depth-registered image data. Columns can be added, sliced, plotted, iterated over in chunks, saved, and loaded (in either single-file ``.pkl`` format or multi-file ``.npy`` format).

For a demonstration of the ``CoreColumn`` API, see: `notebooks/column_demo.py <https://github.com/rgmyr/corebreakout/blob/master/notebooks/column_demo.ipynb>`__

.. toctree::
   :maxdepth: 2
   :caption: Usage Documentation:

   creating_datasets
   model_building
   layout_parameters
   scripts_reference


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |workflowfigure| image:: images/JOSS_figure_workflow.png
