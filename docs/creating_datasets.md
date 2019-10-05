# Using and Adding `Datasets`

## `corebreakout.datasets.PolygonDataset`

This is a subclass of `mrcnn.utils.Dataset` for instance segmentation annotations in the default JSON format of [wkentaro/labelme](https://github.com/wkentaro/labelme). Used for training and validation testing of `mrcnn.model.MaskRCNN` models.

- Assumes `<fname>.jpeg` images in a flat directory with corresponding `<fname>.json` annotations
- Instances of a class are differentiated after the class name (*e.g.*, `column1`, `column2`)

## Usage

```
from corebreakout.datasets import PolygonDataset

data_dir = defaults.DEFAULT_DATA_DIR    # parent of any separate annotation data directories
subset = 'train'                        # subdirectory to read from

dataset = PolygonDataset(classes=defaults.DEFAULT_CLASSES)
dataset.collect_annotated_images(data_dir, subset)
dataset.prepare()
print(dataset)
```

## Writing your own `Dataset`

If you want to use a different annotation format, you can inherit from the base `mrcnn.utils.Dataset` class. It stores

- `collect_annotated_images(data_dir, subset)`
  - Add all of images from a directory and subset (subdirectory) to the `Dataset` via the `add_image()` method.    
- `load_mask(image_id)`
  - Given an `image_id`, load (and compute, if necessary) the corresponding mask. For a mask with `N` object instances (not including the background), the return value from this function is `(mask, class_ids)`, where `mask` is boolean array of shape `(H,W,N)` and `class_ids` is an 1D integer array of size `N` with one `class_id` for each of the instance channels in `mask`.
- `image_reference(image_id)`
  - Return the path of an image, a link to it, or some other details about it that help in looking it up or debugging it. See the code for an example.

## Documentation for `mrcnn.utils.Dataset`

**Attributes:**
- `@property image_ids`
- `image_info`
  - List of `dict` info for each image, each having at least `'source'`, `'id'`, and `'path'` keys
- `class_info`
  - List of classes, with each represented as a `dict` with keys `'source'`, `'id'`, and `'name'`. Background is always the first class.
- ``

**Methods:**
- `add_class(source, class_id, class_name)`
- `add_image`
- `image_reference`
- `prepare()`
- `map_source_class`
- `source_image_link`
- `load_image`
- `load_mask`

To subclass, you can write some user-called method to collect file information:
- `collect_annotated_images(data_dir, subset)`: register the `image_id`, `path`, and `ann_path`

And then ovverride the methods:
- `load_mask(image_id)`
  - Given an `image_id`, load (and compute, if necessary) the corresponding mask. For a mask with `N` object instances (not including the background), the return value from this function should be `(mask, class_ids)`, where `mask` is boolean array of shape `(H,W,N)` and `class_ids` is an 1D integer array of size `N` with one `class_id` for each of the channels in `mask`.
- `image_reference(image_id)`
  - Return the path of an image, a link to it, or some unique property to help in looking it up or debugging it.
