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

import datetime as dt
import unittest
from unittest.mock import MagicMock, Mock
import math
import os

import numpy as np
import pandas as pd
import h5py
from matplotlib.colors import Normalize
import pytz

from utilities import xycoord, prog
from txm.xanes_frameset import XanesFrameset, calculate_whiteline
from txm.frame import (
    average_frames, TXMFrame, xy_to_pixel, pixel_to_xy, Extent, Pixel,
    rebin_image, apply_reference)
from txm.edges import Edge, k_edges
from txm.importers import import_txm_framesets
from txm.xradia import decode_ssrl_params, XRMFile
from txm import xanes_frameset
from txm import plotter

testdir = os.path.join(os.path.dirname(__file__), 'testdata')
ssrldir = os.path.join(testdir, 'ssrl-txm-data')
apsdir = os.path.join(testdir, 'aps-txm-data')

class HDFTestCase(unittest.TestCase):
    """A test case that sets up and tears down an HDF file."""
    def setUp(self):
        curdir = os.path.dirname(os.path.realpath(__file__))
        self.hdf_filename = os.path.join(curdir, 'txm-frame-test.hdf')
        if os.path.exists(self.hdf_filename):
            os.remove(self.hdf_filename)
        self.hdf_file = h5py.File(self.hdf_filename, 'w-')

    def tearDown(self):
        self.hdf_file.close()
        if os.path.exists(self.hdf_filename):
            os.remove(self.hdf_filename)


class XrayEdgeTest(unittest.TestCase):
    def setUp(self):
        self.edge = Edge(
            (8250, 8290, 20),
            (8290, 8295, 1),
            pre_edge=(8250, 8290),
            post_edge=(8290, 8295),
            map_range=(8291, 8293),
            name="Test edge",
        )
    def test_energies(self):
        self.assertEqual(
            self.edge.all_energies(),
            [8250, 8270, 8290, 8291, 8292, 8293, 8294, 8295]
        )

    def test_norm_energies(self):
        self.assertEqual(
            self.edge.energies_in_range(),
            [8291, 8292, 8293]
        )


class TXMMapTest(HDFTestCase):
    def setUp(self):
        ret = super().setUp()
        # Create an HDF Frameset for testing
        self.fs = XanesFrameset(filename=self.hdf_filename,
                                groupname='mapping-test')
        for i in range(0, 3):
            frame = TXMFrame()
            frame.energy = i + 8000
            ds = np.zeros(shape=(3, 3))
            ds[:] = i + 1
            frame.image_data = ds
            self.fs.add_frame(frame)
        self.fs[1].image_data.write_direct(np.array([
            [0, 1, 4],
            [1, 2.5, 1],
            [4, 6, 0]
        ]))
        return ret

    def test_max_energy(self):
        expected = [
            [8002, 8002, 8001],
            [8002, 8002, 8002],
            [8001, 8001, 8002]
        ]
        result = self.fs.whiteline_map()
        self.assertTrue(np.array_equal(result, expected))


class TXMMathTest(unittest.TestCase):
    """Holds tests for functions that perform base-level calculations."""
    def test_calculate_whiteline(self):
        absorbances = [700, 705, 703]
        energies = [50, 55, 60]
        data = pd.Series(absorbances, index=energies)
        out = calculate_whiteline(data)
        self.assertEqual(out, 55)
        # Test using multi-dimensional absorbances (eg image frames)
        absorbances = [np.array([700, 700]),
                       np.array([705, 703]),
                       np.array([703, 707])]
        data = pd.Series(absorbances, index=energies)
        out = calculate_whiteline(data)
        self.assertTrue(np.array_equal(out, [55, 60]))


class TXMImporterTest(unittest.TestCase):
    def setUp(self):
        self.hdf = os.path.join(ssrldir, 'testdata.hdf')
        # Disable progress bars
        prog.quiet = True

    def tearDown(self):
        if os.path.exists(self.hdf):
            os.remove(self.hdf)

    def test_import_timestamp(self):
        fs, = import_txm_framesets(ssrldir, hdf_filename=self.hdf, flavor='ssrl')
        frame = fs[0]
        self.assertEqual(
            frame.starttime,
            dt.datetime(2015, 2, 22, 10, 47, 19, tzinfo=pytz.timezone('US/Pacific'))
        )
        self.assertEqual(
            frame.endtime,
            dt.datetime(2015, 2, 22, 10, 47, 26, 500000, tzinfo=pytz.timezone('US/Pacific'))
        )
        # Verify that the frameset finds the full range of timestamps
        self.assertEqual(
            fs.starttime(),
            dt.datetime(2015, 2, 22, 10, 46, 31, tzinfo=pytz.timezone('US/Pacific'))
        )
        self.assertEqual(
            fs.endtime(),
            dt.datetime(2015, 2, 22, 10, 47, 26, 500000, tzinfo=pytz.timezone('US/Pacific'))
        )

class TXMFrameTest(HDFTestCase):

    def test_average_frames(self):
        # Define three frames for testing
        frame1 = TXMFrame()
        frame1.image_data = np.array([
            [1, 2, 3],
            [3, 4, 5],
            [2, 5, 7],
        ])
        frame2 = TXMFrame()
        frame2.image_data = np.array([
            [3, 5, 7],
            [7, 9, 11],
            [5, 11, 15],
        ])
        frame3 = TXMFrame()
        frame3.image_data = np.array([
            [7, 11, 15],
            [15, 19, 23],
            [11, 23, 31],
        ])
        avg_frame = average_frames(frame1, frame2, frame3)
        expected_array = np.array([
            [11/3, 18/3, 25/3],
            [25/3, 32/3, 39/3],
            [18/3, 39/3, 53/3],
        ])
        # Check that it returns an array with same shape
        self.assertEqual(
            frame1.image_data.shape,
            avg_frame.image_data.shape
        )
        # Check that the averaging is correct
        self.assertTrue(np.array_equal(avg_frame.image_data, expected_array))

    def test_params_from_ssrl(self):
        # First a reference frame
        ref_filename = "rep01_000001_ref_201511202114_NCA_INSITU_OCV_FOV01_Ni_08250.0_eV_001of010.xrm"
        result = decode_ssrl_params(ref_filename)
        expected = {
            'date_string': '',
            'sample_name': 'NCA_INSITU_OCV_FOV01_Ni',
            'position_name': '',
            'is_background': True,
            'energy': 8250.0,
        }
        self.assertEqual(result, expected)
        # Now a sample field of view
        sample_filename = "rep01_201511202114_NCA_INSITU_OCV_FOV01_Ni_08250.0_eV_001of010.xrm"
        result = decode_ssrl_params(sample_filename)
        expected = {
            'date_string': '',
            'sample_name': 'NCA_INSITU_OCV_FOV01_Ni',
            'position_name': '',
            'is_background': False,
            'energy': 8250.0,
        }
        self.assertEqual(result, expected)

    def test_timestamp_from_xrm(self):
        sample_filename = "rep01_201502221044_NAT1050_Insitu03_p01_OCV_08250.0_eV_001of002.xrm"
        xrm = XRMFile(os.path.join(ssrldir, sample_filename), flavor="ssrl")
        # Check start time
        start = dt.datetime(2015, 2, 22, 10, 47, 19, tzinfo=pytz.timezone('US/Pacific'))
        self.assertEqual(xrm.starttime(), start)
        # Check end time (offset determined by exposure time)
        end = dt.datetime(2015, 2, 22, 10, 47, 19, 500000, tzinfo=pytz.timezone('US/Pacific'))
        self.assertEqual(xrm.endtime(), end)

        # Test APS frame
        sample_filename = "20151111_UIC_XANES00_sam01_8313.xrm"
        xrm = XRMFile(os.path.join(apsdir, sample_filename), flavor="aps")
        # Check start time
        start = dt.datetime(2015, 11, 11, 15, 42, 38, tzinfo=pytz.timezone('US/Central'))
        self.assertEqual(xrm.starttime(), start)
        # Check end time (offset determined by exposure time)
        end = dt.datetime(2015, 11, 11, 15, 43, 16, tzinfo=pytz.timezone('US/Central'))
        self.assertEqual(xrm.endtime(), end)

    # def test_magnification_from_xrm(self):
    #     sample_filename = "rep01_201502221044_NAT1050_Insitu03_p01_OCV_08250.0_eV_001of002.xrm"
    #     xrm = XRMFile(os.path.join(ssrldir, sample_filename), flavor="ssrl")
    #     self.assertEqual(xrm.magnification(), 1)

    def test_xy_to_pixel(self):
        extent = Extent(
            left=-1000, right=-900,
            top=300, bottom=250
        )
        result = xy_to_pixel(
            xy=xycoord(x=-950, y=250),
            extent=extent,
            shape=(10, 10)
        )
        self.assertEqual(result, Pixel(vertical=10, horizontal=5))

    def test_pixel_to_xy(self):
        extent = Extent(
            left=-1000, right=-900,
            top=300, bottom=250
        )
        result = pixel_to_xy(
            pixel=Pixel(vertical=10, horizontal=5),
            extent=extent,
            shape=(10, 10)
        )
        self.assertEqual(result, xycoord(x=-950, y=250))

    def test_shift_data(self):
        frame = TXMFrame()
        frame.image_data = self.hdf_file.create_dataset(
            name = 'shifting_data',
            data = np.array([
                [1, 2],
                [5, 7]
            ])
        )
        # Shift in x direction
        frame.shift_data(1, 0)
        expected_data = [
            [2, 1],
            [7, 5]
        ]
        self.assertTrue(np.array_equal(frame.image_data, expected_data))
        # Shift in negative y direction
        frame.shift_data(0, -1)
        expected_data = [
            [7, 5],
            [2, 1]
        ]
        self.assertTrue(np.array_equal(frame.image_data, expected_data))

    def test_rebinning(self):
        frame = TXMFrame()
        original_data = np.array([
            [1., 1., 3., 3.],
            [2, 2, 5, 5],
            [5, 6, 7, 9],
            [8, 12, 11, 10],
        ])
        # Check that binning to same shape return original array
        result_data = rebin_image(original_data, shape=(4, 4))
        self.assertTrue(
            result_data is original_data
        )
        # Check for rebinning by shape
        result_data = rebin_image(original_data, shape=(2, 2))
        expected_data = np.array([
            [6, 16],
            [31, 37]
        ])
        self.assertTrue(np.array_equal(result_data, expected_data))
        # Check for rebinning by factor
        frame.image_data = self.hdf_file.create_dataset(
            name = 'rebinning_data_factor',
            chunks = True,
            data = original_data
        )
        frame.rebin(factor=2)
        self.assertTrue(np.array_equal(frame.image_data, expected_data))
        # Check for error with no arguments
        with self.assertRaises(ValueError):
            frame.rebin()
        # Check for error if trying to rebin to larger shapes
        with self.assertRaisesRegex(ValueError, 'larger than original shape'):
            frame.rebin(factor=0.5)
        with self.assertRaisesRegex(ValueError, 'larger than original shape'):
            frame.rebin(shape=(6, 6))

    def test_subtract_background(self):
        data = np.array([
            [10, 1],
            [0.1, 50]
        ])
        background = np.array([
            [100, 100],
            [100, 100]
        ])
        expected = np.array([
            [1, 2],
            [3, math.log10(2)]
        ])
        result = apply_reference(data, background)
        self.assertTrue(
            np.array_equal(result, expected)
        )
        # Check that uneven samples are rebinned
        data = np.array([
            [3, 1, 0.32, 0],
            [2, 4, 0, 0.68],
            [0.03, -.1, 22, 21],
            [0.07, 0.1, 0, 7],
        ])
        result = apply_reference(data, background)
        self.assertTrue(
            np.array_equal(result, expected)
        )

frameset_testdata = [
    np.array([
        [12, 8, 2.4, 0],
        [9, 11, 0, 1.6],
        [0.12, 0.08, 48, 50],
        [0.09, 0.11, 52, 50],
    ])
]

class MockDataset():
    def __init__(self, value=None):
        self.value = value

    @property
    def shape(self):
        return self.value.shape

class MockFrame(TXMFrame):
    image_data = MockDataset()
    hdf_filename = None
    def __init__(self, *args, **kwargs):
        pass


class MockFrameset(XanesFrameset):
    hdf_filename = None
    parent_groupname = None
    active_particle_idx = None
    edge = k_edges['Ni']
    def __init__(self, *args, **kwargs):
        pass

    def normalizer(self):
        return Normalize(0, 1)

    def __len__(self):
        return len(frameset_testdata)

    def __iter__(self):
        for d in frameset_testdata:
            frame = MockFrame()
            frame.image_data.value = d
            yield frame

# class TXMGtkViewerTest(unittest.TestCase):
#     @unittest.expectedFailure
#     def test_background_frame(self):
#         from txm import gtk_viewer
#         fs = MockFrameset()
#         viewer = gtk_viewer.GtkTxmViewer(frameset=fs,
#                                          plotter=plotter.DummyGtkPlotter(frameset=fs))

if __name__ == '__main__':
    unittest.main()
