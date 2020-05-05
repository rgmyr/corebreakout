.. _model-building:

Building Mask R-CNN Models
==========================

For best results, most users will want to train models on some of
their own data. See :ref:`creating-datasets` for guidelines.

We may release a more general pretrained model in the future, pending demand and the availability of open source datasets. Feel free to contact us if you would like to contribute your data for that purpose.

The rough outline of model construction and training looks like this:

.. code:: python

   import mrcnn.model as modellib
   from corebreakout import defaults, datasets

   model_config = defaults.DefaultConfig()

   train_dataset = datasets.PolygonDataset(...)
   test_dataset = datasets.PolygonDataset(...)

   model = modellib.MaskRCNN(...)

   model.train(train_dataset, test_dataset, ...)

For the finer details see `scripts/train_mrcnn_model.py <https://github.com/rgmyr/corebreakout/blob/master/scripts/train_mrcnn_model.py>`__

``mrcnn`` Model Configuration
-----------------------------

Models are created with a subclass of ``mrcnn.config.Config``. See
``corebreakout.defaults.DefaultConfig`` for our latest model
configuration.

To see all available configuration parameters, see
`mrcnn/config.py <https://github.com/matterport/Mask_RCNN/blob/3deaec5d902d16e1daf56b62d5971d428dc920bc/mrcnn/config.py>`__

The obvious parameters that a user might want to change include:

  - ``NAME``
  - ``RPN_ANCHOR_RATIOS`` : defaults are set up for horizontal columns. Something like ``[1.0, 3.0, 7.0]`` would make more sense for vertical columns.
  - ``STEPS_PER_EPOCH``, ``VALIDATION_STEPS`` : batches per training epoch and validation step. Does not necessarily need to match dataset sizes.
  - ``IMAGES_PER_GPU`` : you can try to increase if you have a large GPU.
  - ``GPU_COUNT`` : you can increase if you multiple GPUs.

You can either modify ``DefaultConfig`` directly, or instantiate your own ``Config`` subclass.

Model Training
--------------

The simplest way to train a model is by running
`scripts/train_mrcnn_model.py <https://github.com/rgmyr/corebreakout/blob/master/scripts/train_mrcnn_model.py>`__.
This script loads pretrained ``COCO`` weights, and executes a three step
training + tuning run.

It also serves as a demonstration of ``Dataset`` collection, and
instantiating and training a ``mrcnn.modellib.MaskRCNN``.

While training, ``mrcnn`` logs ``tensorboard`` files to the specified
``model_dir``. You can view the files by running:

::

   $ tensorboard --logdir <model_dir>

Model Selection
---------------

We recommend viewing the ``tensorboard`` files (and particularly the
``val_loss`` scalar) to select candidate models.

`notebooks/select_model.ipynb <https://github.com/rgmyr/corebreakout/blob/master/notebooks/select_model.ipynb>`__
provides a template for viewing the output of candidate models on the
test dataset.

**Note**: ``mrcnn`` saves Checkpoints each epoch starting at ``0001``,
while ``tensorboard`` logs epochs starting from ``0``. So, if epoch
``X`` looks good on ``tensorboard``, you will want to reference epoch
``X+1`` in your list of candidates to load the corresponding weights.

Using a Model
-------------

Once you have trained and selected a new model, you may want to change
the ``*PATH`` variables in ``corebreakout/defaults.py`` to point to the location of the
new model weights (these paths are what get referenced by default in
``scripts/process_directory.py``, etc.).

Alternatively, you can always pass whatever ``model_dir`` and ``weights_path`` (and
``model_config`` instance) you like when constructing a ``CoreSegmenter``.
