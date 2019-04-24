# `corebreakout.datasets`

We define a couple of subclasses of `mrcnn.utils.Dataset` specifically for core photo datasets with instance segmentation labels. Instances of these objects are used for training and validation testing of `mrcnn.model.MaskRCNN` models:

## Usage of our `Dataset` classes

```
from corebreakout.datasets import PolygonDataset

data_dir = DEFAULT_DATA_DIR    # parent of annotation data directories
subset = 'train'               # subdirectory to read from

dataset = PolygonDataset(classes=DEFAULT_CLASSES)
dataset.collect_annotated_images(data_dir, subset)
dataset.prepare()
print(dataset)
```

## Writing your own custom `Dataset`

We hope that classes provided are sufficient for most use of `corebreakout`, but some users may want to write custom `Dataset` classes, particularly if they have additional labeled images with a different annotation format. Typically this will require inheriting from the base `mrcnn.utils.Dataset` class and overriding a few of its methods:

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
