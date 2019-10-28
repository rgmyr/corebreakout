#!/usr/bin/env python3
import os
from setuptools import find_packages

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

setup(name='corebreakout',
      version='0.2',
      description='Image segmentation tools for processing geological core sample images',
      url='https://github.com/rgmyr/corebreakout',
      author='Ross Meyer',
      author_email='ross.meyer@utexas.edu',
      packages=find_packages(PACKAGE_PATH),
      install_requires=[
            'numpy >= 1.13.0',
            'scikit-image >= 0.13.1',
            'matplotlib',
            
      ],
      zip_safe=False
)
