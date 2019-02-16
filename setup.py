#!/usr/bin/env python3
'''
Seems to work fine, might update later.
'''

import os
from setuptools import find_packages

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

setup(name='corebreakout',
      version='0.1',
      description='Tools for segmentation and labeling of core sample images',
      url='https://github.com/rgmyr/corebreakout',
      author='Ross Meyer',
      author_email='ross.meyer@utexas.edu',
      packages=find_packages(PACKAGE_PATH),
      install_requires=[
            'numpy >= 1.13.0',
            'scipy >= 1.0.0',
            'scikit-image >= 0.13.1',
            'matplotlib'
      ],
      zip_safe=False
)
