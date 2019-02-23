
Utilities and scripts for generating and handling Pascal-VOC style XML data labels (create labels with e.g., `LabelImg`).

### `split_npy_image.py`

Takes `image.npy` and `depth.npy` files from `src`, writes a set of images (under jpeg size limit) to `<dst>/well/`.

### `join_xml_labels.py`

Concatenates XML label files from a `src` directory, writes a row labels file (required for modeling) to `dst`.
