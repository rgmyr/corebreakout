# Model Building

For best results, most users will want to train models on their own datasets. We may consider releasing a more general pretrained model in the future.

## `mrcnn` Model Configuration

Models are created with a subclass of `mrcnn.config.Config`. See `corebreakout.defaults.DefaultConfig` for our latest model configuration. The parameters that a user might want to change include:
- `NAME`
- `RPN_ANCHOR_RATIOS` : defaults are set up for horizontal columns. Something like `[1.0, 3.0, 7.0]` would make more sense for vertical columns.
- `STEPS_PER_EPOCH`, `VALIDATION_STEPS` : images (batches?) per training epoch and validation step.
- `IMAGES_PER_GPU` : you can increase if you have multiple GPUs.
- `GPU_COUNT` : you can try to increase if you have a large GPU.

You can either modify `DefaultConfig`, or create your own `Config` subclass.

## Model Training

The easiest way to train a model is by running `scripts/train_mrcnn_model.py`. This script loads pretrained `COCO` weights.
