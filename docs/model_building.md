# Model Building

For the best results, most users will want to train models on some of their own data. See `creating_datasets.md`.

We may release a more general pretrained model in the future, pending demand and the availability of open source datasets.

## `mrcnn` Model Configuration

Models are created with a subclass of `mrcnn.config.Config`. See `corebreakout.defaults.DefaultConfig` for our latest model configuration.

To see all available configuration parameters, see [mrcnn/config.py](https://github.com/matterport/Mask_RCNN/blob/3deaec5d902d16e1daf56b62d5971d428dc920bc/mrcnn/config.py)

The obvious parameters that a user might want to change include:
- `NAME`
- `RPN_ANCHOR_RATIOS` : defaults are set up for horizontal columns. Something like `[1.0, 3.0, 7.0]` would make more sense for vertical columns.
- `STEPS_PER_EPOCH`, `VALIDATION_STEPS` : images (batches?) per training epoch and validation step.
- `IMAGES_PER_GPU` : you can try to increase if you have a large GPU.
- `GPU_COUNT` : you can increase if you multiple GPUs.

You can either modify `DefaultConfig`, or create your own `Config` subclass.

## Model Training

The easiest way to train a model is by running `scripts/train_mrcnn_model.py`. This script loads pretrained `COCO` weights, and executes a three step training + tuning run.

While training, `mrcnn` logs `tensorboard` files to the specified `model_dir`. You can view the files by running:

```
$ tensorboard --logdir <model_dir>
```
