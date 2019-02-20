"""
Dataset class for COCO format segmentation labels
"""
import os
import glob
import skimage
import numpy as np

from mrcnn.utils import Dataset
from mrcnn.config import Config
import mrcnn.model as modellib

from corebreakout.config import DEFAULT_CLASSES



#DEFAULT_CLASSES = ['col', 'box', 'scale']


class PolygonDataset(Dataset):

    def __init__(self, classes=DEFAULT_CLASSES):

        super().__init__()

        for i, cls_name in zip(range(len(classes)), classes):
            # `source` doesn't matter for us, just saying 'cb' for 'corebreakout'
            self.add_class('cb', i+1, cls_name)     # Note: 'BG' = class 0


    def collect_annotated_images(self, data_dir, subset):
        """
        Check for annotation ('.json') and image ('.jpg'/'.jpeg') pairs,
        and add them all to the dataset. Must be at least one pair in `data_dir`/`subset`.
        """
        data_dir = Path(data_dir) / subset
        assert data_dir.is_dir(), f'Directory {data_dir} must exist.'

        annotations = sorted(data_dir.glob('*.json'))
        assert len(annotations), 'Must be at least one annotation file in `data_dir`'

        for ann_path in annotations:
            image_matches = list(data_dir.glob(ann_path.stem + '*.jp*g'))
            try:
                img_path = image_matches[0]
            except IndexError:
                raise UserWarning(f'Matching .jpg/.jpeg not found for {ann_path}')

            self.add_image('cb', image_id=ann_path.stem, path=img_path, ann_path=ann_path)


    def load_mask(self, image_id):
        """
        Return the instance `mask` and `class_ids` arrays for a given `image_id`.
        """
        ann_path = self.image_info[image_id]['ann_path']

        with open(ann_path, 'r') as ann_file:
            ann_json = json.load(ann_file)

        return self.ann_to_mask(ann_json)


    def ann_to_mask(self, ann):
        """
        Take an annotation dict, return `(mask, class_ids)` arrays.
        Assumes that some classes may have multiple labels (e.g., 'col1', 'col2', ...),
        and that each label may include multiple polygons.
        """
        h, w = ann['imageHeight'], ann['imageWidth']

        unique_labels = list(set([s['label'] for s in ann['shapes']]))

        masks = np.zeros((h, w, len(unique_labels)), dtype=np.bool)

        class_ids = np.array([self.label_to_class_id(l) for l in unique_labels])

        for polygon in ann['shapes']:
            channel = unique_labels.index(polygon['label'])
            coords = np.array(polygon['points'])
            rr, cc = skimage.draw.polygon(coords[:,1], coords[:,0])
            masks[rr,cc,channel] = True

        return masks, class_ids


    def label_to_class_id(self, label):
        """
        Return `class_id` corresponding to `label` given that `label` just needs to start with the class name.
        """
        matches = [label.startswith(c['name']) for c in self.class_info]

        assert any(matches), f'Label {label} must match a class from classes: {self.class_info}'
        assert sum(matches) == 1, f'Label {label} cant match multiple classes in {self.class_info}'

        return self.class_info[matches.index(True)]['id']


    def image_reference(self, image_id):
        """
        Return the path of the image corresponding to `image_id`, if there is one.
        """
        if self.image_info[image_id]['source'] == 'cb':
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


    def __repr__(self):
        return(
            f'\n PolygonDataset\n'
            f'Image count : {self.num_images}\n'
            f'Class count : {self.num_classes}\n'
            '\n'.join(['{:3}. {:50}'.format(i, info['name']) for i, info in enumerate(self.class_info))
        )
