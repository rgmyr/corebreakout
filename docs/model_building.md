# Model Building

# `mrcnn` Model Configuration

Models are created with a subclass of `mrcnn.config.Config`. See `corebreakout.defaults.DefaultConfig` for our latest model configuration. The parameters that you might obviously need to change are:
- `NAME`
- `RPN_ANCHOR_RATIOS` : defaults are set up for horizontal columns. Something like `[1.0, 3.0, 7.0]` would make sense for vertical columns.
- `STEPS_PER_EPOCH`, `VALIDATION_STEPS` : images (batches) per training epoch and validation step.
- `IMAGES_PER_GPU` : you can increase if you have multiple GPUs.
- `GPU_COUNT` : you can try to increase if you have a large GPU.
