# -*- coding: utf-8 -*-
#
# Copyright © 2018 Mark Wolfman
#
# This file is part of Scimap.
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

# flake8: noqa


import unittest
import os
import shutil
import math
import warnings

import h5py
import numpy as np

from scimap.coordinates import Cube
from scimap import gadds
from scimap.importers import import_gadds_map

TESTDIR = os.path.dirname(__file__)


class GaddsTest(unittest.TestCase):
    """Test how the software interacts with Bruker's GADDS control
    systems, both importing and exporting."""
    hdf_filename = os.path.join(TESTDIR, "temp_map_gadds.h5")
    hdf_groupname = "test_map_gadds"
    directory = os.path.join(TESTDIR, 'test-data-xrd', 'xrd-map-gadds')
    
    def setUp(self):
        # Write the script to ensure HDF5 file exists
        gadds.write_gadds_script(two_theta_range=(55, 75),
                                 sample_name=self.hdf_groupname,
                                 diameter=1.2,
                                 center=(0, 0), hdf_filename=self.hdf_filename,
                                 hexadecimal=False)
    
    def tearDown(self):
        try:
            os.remove(self.hdf_filename)
        except FileNotFoundError:
            pass
        gadds_dir = os.path.join(TESTDIR, self.hdf_groupname + '-frames')
        if os.path.exists(gadds_dir):
            shutil.rmtree(gadds_dir)
    
    def test_import_gadds_map(self):
        import_gadds_map(directory=self.directory,
                         hdf_filename=self.hdf_filename,
                         hdf_groupname=self.hdf_groupname)
        # Check that the hdf5 file was created
        self.assertTrue(os.path.exists(self.hdf_filename))
        # Check that intensities were saved
        with h5py.File(self.hdf_filename) as f:
            keys = list(f[self.hdf_groupname].keys())
        self.assertIn("intensities", keys)
        self.assertIn("two_thetas", keys)


class SlamFileTest(unittest.TestCase):
    def tearDown(self):
        # Remove temporary files created when writing slam files.
        try:
            os.remove(os.path.join(TESTDIR, 'xrd-map-gadds-temp.h5'))
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree('xrd-map-gadds-temp-frames')
        except FileNotFoundError:
            pass
    
    def test_number_of_frames(self):
        num_frames = gadds._number_of_frames(two_theta_range=(10, 80), frame_step=20)
        self.assertEqual(num_frames, 4)
        # Check for values outside of limits
        with self.assertRaises(ValueError):
            gadds._number_of_frames(two_theta_range=(60, 180), frame_step=20)
    
    # def test_rows(self):
        # Does passing a resolution set the appropriate number of rows
        # self.sample = XRDMap(diameter=12.7, resolution=0.5, qrange)
        # self.assertEqual(
        #     self.sample.rows,
        #     18
        # )
    
    def test_detector_start(self):
        self.assertEqual(
            gadds._detector_start(two_theta_range=(50, 90), frame_width=20),
            10
        )
    
    def test_source_angle(self):
        theta1 = gadds._source_angle(two_theta_range=(50, 90))
        self.assertEqual(
            theta1,
            50
        )
        # Check for values outside the limits
        with self.assertRaises(ValueError):
            gadds._source_angle(two_theta_range=(-5, 50))
        # Check for values outside theta1 max
        theta1 = gadds._source_angle(two_theta_range = (60, 90))
        self.assertEqual(theta1, 50)
    
    @unittest.expectedFailure
    def test_small_angles(self):
        """
        See what happens when the 2theta angle is close to the max X-ray
        source angle.
        """
        self.sample.two_theta_range = (47.5, 62.5)
        self.assertEqual(
            self.sample.get_theta1(),
            47.5
        )
        self.assertEqual(
            self.sample.get_theta2_start(),
            10
        )
    
    def test_path(self):
        results_list = []
        for coords in gadds._path(2):
            results_list.append(coords)
        self.assertEqual(
            results_list,
            [Cube(0, 0, 0),
             Cube(1, 0, -1),
             Cube(0, 1, -1),
             Cube(-1, 1, 0),
             Cube(-1, 0, 1),
             Cube(0, -1, 1),
             Cube(1, -1, 0)]
        )
    
    @unittest.expectedFailure
    def test_coverage(self):
        halfMap = XRDMap(collimator=2, coverage=0.25)
        self.assertEqual(halfMap.unit_size, 2 * math.sqrt(3))
    
    @unittest.expectedFailure
    def test_cell_size(self):
        unitMap = XRDMap(collimator=2)
        self.assertEqual(unitMap.unit_size, math.sqrt(3))
    
    def test_jinja_context(self):
        context = gadds._context(diameter=10, collimator=0.5,
                                 coverage=1, scan_time=300,
                                 two_theta_range=(10, 20),
                                 detector_distance=20,
                                 frame_size=1024, center=(-10.5, 20.338),
                                 sample_name="xrd-map-gadds-temp",
                                 hexadecimal=False)
        self.assertEqual(len(context['scans']), 631)
        self.assertAlmostEqual(context['scans'][1]['x'], 0.2165, places=4)
        self.assertAlmostEqual(context['scans'][1]['y'], 0.375)
        self.assertEqual(context['scans'][3]['filename'], 'map-3')
        self.assertEqual(context['xoffset'], -10.5)
        self.assertEqual(context['yoffset'], 20.338)
        # Flood and spatial files to load
        self.assertEqual(context['flood_file'], '1024_020._FL')
        self.assertEqual(context['spatial_file'], '1024_020._ix')
    
    def test_write_slamfile(self):
        sample_name = "xrd-map-gadds-temp"
        directory = '{}-frames'.format(sample_name)
        hdf_filename = os.path.join(TESTDIR, "{}.h5".format(sample_name))
        # Check that the directory does not already exist
        self.assertFalse(
            os.path.exists(directory),
            'Directory {} already exists, cannot test'.format(directory)
        )
        # Write the slamfile
        gadds.write_gadds_script(two_theta_range=(55, 70),
                                 sample_name=sample_name,
                                 center=(0, 0), hdf_filename=hdf_filename)
        # Test if the correct things were created
        self.assertTrue(os.path.exists(directory))
        slamfile = os.path.join(directory, sample_name + ".slm")
        self.assertTrue(os.path.exists(directory))
        # Test that the HDF5 file was created to eventually hold the results
        self.assertTrue(os.path.exists(hdf_filename))
        with h5py.File(hdf_filename) as f:
            keys = list(f[sample_name].keys())
            self.assertIn('positions', keys)
            layout = f[sample_name]['positions'].attrs['layout']
            self.assertEqual(layout, 'hex')
            self.assertIn('file_basenames', keys)
            wavelength = f[sample_name]['wavelengths']
            print(wavelength)
            np.testing.assert_almost_equal(wavelength,
                                           [[1.5406, 1],
                                            [1.5444, 0.5]])
            # self.assertEqual(wavelength[0], 1.5406)
            # self.assertEqual(wavelength[1], 1.5444)
            # Collimator and step_sizes
            self.assertEqual(f[sample_name]['collimator'].value, 0.8)
            self.assertEqual(f[sample_name]['collimator'].attrs['unit'], 'mm')
            self.assertEqual(f[sample_name]['step_size'].value, math.sqrt(3) * 0.8 / 2)
        # Check that writing a second slam file does not overwrite the original data
        with warnings.catch_warnings() as w:
            gadds.write_gadds_script(two_theta_range=(55, 70),
                                     sample_name=sample_name,
                                     center=(0, 0), hdf_filename=hdf_filename,
                                     overwrite=False)
            print(w)
