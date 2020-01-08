#!/usr/bin/env python3
import os
from setuptools import find_packages

try:
    from setuptools import setup
except ImportError:
    raise UserWarning('`distutils` is not supported since you must use Python>=3.6')


PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))


# Mostly a duplication of requirements.txt
# with the addition of pip-only package `imgaug`
install_requires = [
    'numpy<=1.16.4',
    'scipy',
    'dill',
    'Pillow',
    'cython',
    'matplotlib',
    'scikit-image',
    'keras>=2.0.8,<=2.2.5',
    'opencv-python',
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
