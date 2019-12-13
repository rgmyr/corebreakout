"""
Dataset class for COCO format segmentation labels
"""
import os
import json
from pathlib import Path
from itertools import product

import skimage
import numpy as np

from mrcnn.utils import Dataset

from corebreakout import defaults


class PolygonDataset(Dataset):
    """Subclass of `mrcnn.utils.Dataset` for polygonal JSON annotations in `labelme` format.
    See `wkentaro/labelme` to get the GUI. Outputs a JSON file with list of polygon 'shapes'.

    Labels must start with a unique class name, but instances can be differentiated afterward
    however you like. For example, different `col` instances can be labeled 'col1', 'col2', etc.
    And multiple polygons may belong to a single instance of a class.

    The tradeoff is that no class name can a substring of any other class name.
    """

    def __init__(self, classes=defaults.CLASSES):
        super().__init__()

        if not self.check_classes(classes):
            raise ValueError(f"{classes} are invalid.")

        for i, cls_name in zip(range(len(classes)), classes):
            # `source` doesn't matter for single dataset, just using 'cb' for 'corebreakout'
            self.add_class("cb", i + 1, cls_name)  # Note: 'BG' = class 0

    def collect_annotated_images(self, data_dir, subset):
        """Check for annotation ('.json') and image ('.jpg'/'.jpeg') pairs, and add them.

        Corresponding annotation and image paths should differ only in their file extensions.
        """
        data_dir = Path(data_dir) / subset
        assert data_dir.is_dir(), f"Directory {data_dir} must exist."

        annotations = sorted(data_dir.glob("*.json"))
        assert len(
            annotations
        ), "There must be at least one annotation file in `data_dir`"

        for ann_path in annotations:
            image_matches = list(data_dir.glob(ann_path.stem + "*.jp*g"))
            try:
                img_path = image_matches[0]
                self.add_image(
                    "cb", image_id=ann_path.stem, path=img_path, ann_path=ann_path
                )
            except IndexError:
                raise UserWarning(f"Matching .jpg/.jpeg not found for {ann_path}")

    def load_mask(self, image_id):
        """Return the `mask` and `class_ids` arrays for a given `image_id`."""
        ann_path = self.image_info[image_id]["ann_path"]

        with open(ann_path, "r") as ann_file:
            ann_json = json.load(ann_file)

        return self.ann_to_mask(ann_json)

    def ann_to_mask(self, ann):
        """Take JSON annotation dict, return `(mask, class_ids)` arrays.

        Assumes that some classes may have multiple instances ('col1', 'col2', etc.),
        and that each labeled instance may be composed of multiple polygons.
        """
        unique_labels = list(set([p["label"] for p in ann["shapes"]]))

        class_ids = np.array([self.label_to_class_id(l) for l in unique_labels])

        h, w = ann["imageHeight"], ann["imageWidth"]
        masks = np.zeros((h, w, len(unique_labels)), dtype=np.bool)

        for polygon in ann["shapes"]:
            boundary = np.array(polygon["points"])
            rr, cc = skimage.draw.polygon(boundary[:, 1], boundary[:, 0])
            instance = unique_labels.index(polygon["label"])
            masks[rr, cc, instance] = True

        return masks, class_ids

    def label_to_class_id(self, label):
        """
        Return `class_id` corresponding to `label` given that `label` just needs to start with the class name.
        """
        matches = [label.startswith(c["name"]) for c in self.class_info]

        assert any(
            matches
        ), f"Label {label} must match a class from classes: {self.class_info}"
        assert (
            sum(matches) == 1
        ), f"Label {label} cant match multiple classes in {self.class_info}"

        return self.class_info[matches.index(True)]["id"]

    def image_reference(self, image_id):
        """Return the path of the image corresponding to `image_id`, if there is one."""
        info = self.image_info[image_id]
        if info["source"] == "cb":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def __repr__(self):
        return (
            f"\n PolygonDataset\n"
            f"Image count : {len(self.image_ids)}\n"
            f"Class count : {self.num_classes}\n"
            + "\n".join(
                [
                    "{:3}. {:50}".format(i, info["name"])
                    for i, info in enumerate(self.class_info)
                ]
            )
        )

    @staticmethod
    def check_classes(classes):
        """Make sure no class is a substring of any other class."""
        for pair in product(classes, classes):
            if (pair[0] != pair[1]) and (pair[0] in pair[1]):
                return False
        return True
