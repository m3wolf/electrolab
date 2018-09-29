#!/usr/bin/env python

import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name="scimap",
      version=read('VERSION'),
      description="Tools for analyzing X-ray diffraction mapping data",
      author="Mark Wolf",
      author_email="mark.wolf.music@gmail.com",
      url="https://github.com/m3wolf/scimap",
      keywords="XANES X-ray diffraction operando",
      install_requires=['pytz>=2013b', 'h5py', 'pandas', 'Pillow', 'numpy',
                        'matplotlib', 'scikit-image', 'scikit-learn', 'pint',
                        'Jinja2', 'scipy', 'tqdm'],
      packages=['scimap',],
      # package_data={
      #     'xanespy': ['qt_map_window.ui', 'qt_frame_window.ui']
      # },
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
          'Natural Language :: English',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3.5',
          'Topic :: Scientific/Engineering :: Chemistry',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering :: Visualization',
      ]
)
