#!/usr/bin/env python3
import os
from setuptools import find_packages

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))


# duplication of requirements.txt
install_requires = [
    'numpy<=1.16.4',
    'scipy',
    'dill',
    'Pillow',
    'cython',
    'matplotlib',
    'scikit-image',
    'keras>=2.0.8,<=2.2.5',
    'opencv',
    'h5py',
    'imgaug',
    'IPython[all]'
]


setup(name='corebreakout',
      version='0.2',
      description='Segmentation and depth-alignment of geological core sample images via Mask-RCNN',
      url='https://github.com/rgmyr/corebreakout',
      author='Ross Meyer',
      author_email='ross.meyer@utexas.edu',
      packages=find_packages(PACKAGE_PATH),
      install_requires=install_requires,
      zip_safe=False
)
