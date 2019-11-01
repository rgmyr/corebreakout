
These scripts contain utilities for generating and handling Pascal-VOC style XML data labels (create labels with e.g., `LabelImg`). We have used this scheme to assign interval labels to aggregated `CoreColumn` images, for use in downstream ML pipelines (related code will be released along with a paper at a later date.)

This is entirely separate from `corebreakout` functionality, but may be useful to people interested in undertaking similar projects. Basically, we split saved column image arrays such that they are small enough to be saved as JPEGs (which have a dimension limit of `2^16`). We then assign labels to intervals in `LabelImg`(link here).

### `split_npy_image.py`

Takes `image.npy` and `depth.npy` files from `src`, writes a set of images (under jpeg size limit) to `<dst>/well/`.

### `join_xml_labels.py`

Concatenates XML label files from a `src` directory, writes a row labels file (required for modeling) to `dst`.
