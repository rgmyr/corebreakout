import argparse
import os
import pathlib

import numpy as np
from xml.etree import ElementTree


parser = argparse.ArgumentParser('Join *.xml files into a single array of row labels.')
parser.add_argument('well', type=str, help='Name of well to join labels from. Should name a directory in <src>.')
parser.add_argument('--src', type=str, help='Parent dir of /<well>/*.xml files.',
                    default='/home/'+os.environ['USER']+'/Dropbox/core_data/facies/label/')
parser.add_argument('--dst', type=str, help='Path to write joined row labels array to as .npy.',
                    default='/home/'+os.environ['USER']+'/Dropbox/core_data/facies/train_data/')


XML_NAME = 'FaciesII'
SAVE_NAME = '_labelsII'


class XMLSection():
    """
    Utility class to represent labeled XML sections.
    """
    def __init__(self, xml_obj):
        self.label = xml_obj.find('name').text

        bbox = xml_obj.find('bndbox')
        self.ymin = eval(bbox.find('ymin').text)
        self.ymax = eval(bbox.find('ymax').text)

    def __lt__(self, other):
        """Make sections sortable."""
        return self.ymin < other.ymin


def snap_xml_sections(xml_path):
    """
    Snap XML labels to top and bottom of core, return row labels array.
    Section ymax's get snapped to the ymin of the section below,
    or the end of the array (for the last section in the file).
    """
    tree = ElementTree.parse(xml_path)
    height = eval(tree.find('size').find('height').text)
    label_array = np.zeros((height,), dtype='a2')

    sections = sorted([XMLSection(xobj) for xobj in tree.findall('object')])
    num_sections = len(sections)

    for i, section in enumerate(sections):
        ymin = section.ymin if i > 0 else 0
        ymax = sections[i+1].ymin if i+1 < num_sections else height
        label_array[ymin:ymax] = section.label

    return label_array


def join_xml_labels(well, src_path, dst_path):
    """
    Get all the XML files, snap them, saved concatted array to `dst_path`.
    """
    xml_files = list((src_path / pathlib.Path(well) / XML_NAME).glob('*.xml'))
    assert len(xml_files) > 0, 'At least one XML file must be present.'

    label_arrays = [snap_xml_sections(xml_path) for xml_path in sorted(xml_files)]
    np.save(dst_path / (well + SAVE_NAME + '.npy'), np.concatenate(label_arrays))


if __name__ == '__main__':

    args = parser.parse_args()

    src_path = pathlib.Path(args.src)
    print(f'src: {str(src_path)}')
    dst_path = pathlib.Path(args.dst)
    print(f'dst: {str(dst_path)}')

    assert src_path.is_dir() and src_path.exists(), 'Check src_path'
    assert dst_path.is_dir() and dst_path.exists(), 'Check dst_path'

    join_xml_labels(args.well, src_path, dst_path)
