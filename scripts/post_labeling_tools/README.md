
These scripts contain utilities for generating and handling Pascal-VOC style XML data labels (create labels with e.g., `LabelImg`). We have used this scheme to assign interval labels to aggregated `CoreColumn` images, for use in downstream ML pipelines (related code will be released concurrently with a paper at a later date.)

This is entirely separate from `corebreakout` functionality, but may be useful to people interested in undertaking similar projects (feel free to raise an issue or email ross.meyer<at>utexas.edu for guidance). We may also integrate some of this functionality into the main package (and have started to do so, with *e.g.*, `CoreColumn.iter_chunks()`).

Basically, we split saved column image arrays such that they are small enough to be saved as JPEGs (which have a dimension limit of `2^16`). We then assign labels to intervals in `LabelImg`(link here), and finally snap + concatenate the XML labels and convert to a row-wise labels array.

### `split_npy_image.py`

Takes `image.npy` and `depth.npy` files from `src`, writes a set of images (under jpeg size limit) to `<dst>/well/`.

### `join_xml_labels.py`

Concatenates XML label files from a `src` directory, writes a row labels file (required for modeling) to `dst`.
