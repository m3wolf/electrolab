# -*- coding: utf-8 -*-
#
# Copyright © 2016 Mark Wolf
#
# This file is part of scimap.
#
# Scimap is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Scimap is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Scimap.  If not, see <http://www.gnu.org/licenses/>.

import unittest
import os
import h5py
import sys

# Set backend so matplotlib doesn't try and show plots
import matplotlib
matplotlib.use('Agg')

from scimap import import_gadds_map, exceptions

TEST_DIR = os.path.join(os.path.dirname(__file__), 'test-data-xrd/')
GADDS_DIR = os.path.join(TEST_DIR, 'xrd-map-gadds')
GADDS_HDF = os.path.join(TEST_DIR, 'xrd-map-gadds.h5')
GADDS_NAME = 'xrd-map-gadds'

class ImportGaddsTest(unittest.TestCase):
    def test_no_hdf(self):
        """Does the importer raise an exception if the HDF5 file doesn't exist?"""
        with self.assertRaisesRegex(OSError, "gadds"):
            bad_file = os.path.join(GADDS_DIR, 'gibberish.h5')
            import_gadds_map(sample_name=GADDS_NAME,
                             directory=GADDS_DIR,
                             hdf_filename=bad_file,
                             hdf_groupname=GADDS_NAME)

    def test_hdf_properties(self):
        self.assertTrue(os.path.exists(GADDS_HDF), GADDS_HDF)
        import_gadds_map(sample_name=GADDS_NAME,
                         directory=GADDS_DIR,
                         hdf_filename=GADDS_HDF,
                         hdf_groupname=GADDS_NAME)
        with h5py.File(GADDS_HDF) as f:
            grp = f[GADDS_NAME]
            # self.assertEqual(grp.attrs

if __name__ == '__main__':

    # Look for tests in files in subdirectories
    start_dir = os.path.dirname(__file__)
    tests = unittest.defaultTestLoader.discover(start_dir)
    runner = unittest.runner.TextTestRunner(buffer=False)
    result = runner.run(tests)
    sys.exit(not result.wasSuccessful())
