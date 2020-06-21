Scripts Reference
=================

``train_mrcnn_model.py``
------------------------
Train a new ``mrcnn`` model starting from pretrained COCO weights

optional arguments:
  -h, --help            show this help message and exit
  --steps STEPS         1, 2, or 3. How many of the steps to train: (heads,
                        4+, entire model)
  --model_dir MODEL_DIR
                        Directory in which to create new training
                        subdirectory for checkpoints and tensorboard logs.
  --data_dir DATA_DIR   Directory in which to find ``train`` and ``test``
                        subdirectories containing labeled images.

                        Extract OCR top and base depths from images with `pytesseract`

``get_ocr_depths.py``
---------------------
Extract OCR top and base depths from images with ``pytesseract``

optional arguments:
  -h, --help            show this help message and exit
  --root_dir ROOT_DIR   A common parent directory of all target ``<subdir>`` directories.
  --subdir SUBDIR       A string contained in the name of all target subdirectories.
  --save_name SAVE_NAME
                        Name of depths csv file(s) to be saved in matching subdirs.
  --force               Flag to force overwrite of any existing ``<save_name>.csv`` files.
  --inspect             Flag to inspect images and print OCR output whenever there is an issue.

As an example, you can test the script by running:

.. code:: none

  $ cd scripts
  $ python get_ocr_depths.py --root_dir ../tests/data --subdir two_image_dataset --save_name auto_depths_test

This should save a new file at ``tests/data/two_image_dataset/auto_depths_test.csv``, with contents like:

.. code::

  ,top,bottom
  S00101409.jpeg,2348.0,2350.0
  S00111582.jpeg,7716.0,2220.0

Note that ``7716.0`` is a misread, and should have been ``2218.0``. At least with our BGS images, some manual corrections are usually required, but this provides a template for the ``--depth_csv`` file required to run ``process_directory.py``.


``process_directory.py``
------------------------
Process directory of raw images with Mask R-CNN and save results as a ``CoreColumn``.

The ``path`` given should contain images as jpeg files, and a ``depth_csv`` file in the format:

.. code::

           ,    top,    bottom
  <filename1>, <top1>, <bottom1>
  ...
  <filenameN>, <topN>, <bottomN>

**NOTE**: model ``Config``, ``class_names``, and segmentation ``layout_params`` can only be
changed manually at the top of script, and default to those configured in `defaults.py <https://github.com/rgmyr/corebreakout/blob/master/corebreakout/defaults.py>`_

positional arguments:
  path                 Path to directory of images (and depth information csv) to process.

optional arguments:
 -h, --help            show help message and exit
 --model_dir MODEL_DIR
                       Directory to load ``mrcnn`` model from.
                       Default=``defaults.MODEL_DIR``
 --weights_path WEIGHTS_PATH
                       Path to model weights to load.
                       Default=``defaults.CB_MODEL_PATH``
 --add_tol ADD_TOL     Gap tolerance when adding ``CoreColumn`` objects,
                       default=5.0.
 --add_mode ADD_MODE   ``CoreColumn.add_mode``. One of {'fill', 'collapse'}.
 --depth_csv DEPTH_CSV
                       Name of filename + (top, bottom) csv to read from
                       ``path``, default=``'auto_depths.csv'``
 --save_dir SAVE_DIR   Path to save ``CoreColumn`` to, default=None will save to
                       ``path``
 --save_name SAVE_NAME
                       Name to use for ``CoreColumn.save``, default=None
                       results in ``CoreColumn_<top>_<base>``
 --save_mode SAVE_MODE
                       One of {'pickle', 'numpy'}. Whether to save as single
                       ``pkl`` file or multiple ``npy`` files

Assuming you've downloaded and unzipped the `assets` folder in the default location, you can test the script with default parameters by running:

.. code::

  $ cd scripts
  $ python process_directory.py ../tests/data/two_image_dataset --depth_csv dummy_depths.csv

This should save the aggregated ``CoreColumn`` to ``tests/data/two_image_dataset/CoreColumn_1.00_5.00.pkl``.

``prune_imageData.py``
----------------------
Remove the ``imageData`` field from all JSON files in tree below ``path``:

positional arguments:
  path        Path to parent of all target JSON files.

optional arguments:
  -h, --help  show this help message and exit
