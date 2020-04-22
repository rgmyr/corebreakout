Creating ``Datasets``
=============================

Image Labeling Guidelines
----------------------------------------

The recommended way to add a new set of labeled training images is to
annotate them using
`wkentaro/labelme <https://github.com/wkentaro/labelme>`__. The
``labelme`` GUI allows the user to draw any number of labeled polygons
on an image, and saves the labels and coordinates in a JSON annotation
file.

Simply copy all the images you want to label into a flat directory, open the directory in `labelme`, and begin saving your annotations. To be able to use the built-in `corebreakout.datasets.PolygonDataset` class with your training data, you will want to follow these labeling guidelines:

-  Save ``<fname>.json`` annotations in a flat directory with
   corresponding ``<fname>.jpeg`` files (this is ``labelme``\ ’s default
   behavior)
-  You may label any number of classes. You will have to supply a list
   of these classes to the ``PolygonDataset`` constructor, or modify
   ``defaults.DEFAULT_CLASSES``.
-  Different instances of the same class should begin with the class
   name and be differentiated afterward (*e.g.*, ``col1, col2, col3``)

   -  The corollary is that no class name can be a substring of any
      other class name (*e.g.*, ``col, col_tray`` would not be allowed)
   -  Multiple polygons may belong to a single instance, though we
      recommend keeping masks on the coarser side

-  After annotating images, split into sibling ``'train'`` and
   ``'test'`` directories

**Note:** We've found that the point of diminishing returns happens somewhere in the range of 20-30 training images, which probably corresponds to 30-50 column instances for this dataset. Of course, YMMV.

After compiling the annotations, you may wish to modify
``defaults.DATASET_DIR`` to avoid need to explicitly specify the data
location.

``corebreakout.datasets.PolygonDataset``
----------------------------------------

This is a subclass of ``mrcnn.utils.Dataset`` for instance segmentation
annotations in the default JSON format of
`wkentaro/labelme <https://github.com/wkentaro/labelme>`__.

Usage
~~~~~

::

   from corebreakout.datasets import PolygonDataset

   data_dir = defaults.DEFAULT_DATA_DIR    # parent of any separate annotation data directories
   subset = 'train'                        # which subdirectory to read from

   dataset = PolygonDataset(classes=defaults.DEFAULT_CLASSES)

   # Collect all of the requied ID + path information
   dataset.collect_annotated_images(data_dir, subset)

   # Set all of the attrs required for use
   dataset.prepare()

   print(dataset)

Two `dataset` objects (train, test) are required in calls to `model.train()`, which is why we split them into separate directories.

Subclassing ``mrcnn.utils.Dataset``
-----------------------------------

If you want to use a different annotation format, you can inherit from
the base ``mrcnn.utils.Dataset`` class.

You will need to write some user-called method to collect file
information:

- *e.g.*, ``collect_annotated_images(data_dir, subset)``: Register ``image_id``, ``path``, and ``ann_path`` for each (image, annotation) file pair in ``<data_dir>/<subset>`` directory.

And then override at least these two methods:

- ``load_mask(image_id)``: Given an ``image_id``, load (and compute, if necessary) the corresponding mask. For an with ``N`` objects (not including the background), the return value from this function should be ``(mask, class_ids)``, where ``mask`` is boolean array of shape ``(H,W,N)`` and ``class_ids`` is a 1D integer array of size ``N`` with one ``class_id`` for each channel in ``mask``.
- ``image_reference(image_id)``: Return the path of an image, a link to it, or some other unique property to help in looking it up or debugging it.
