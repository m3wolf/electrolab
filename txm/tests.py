# -*- coding: utf-8 -*-

import unittest
from unittest.mock import MagicMock, Mock
import math
import os

import numpy as np
import h5py
from matplotlib.colors import Normalize

from utilities import xycoord
from txm.xanes_frameset import XanesFrameset
from txm.frame import (
    average_frames, TXMFrame, xy_to_pixel, pixel_to_xy, Extent, Pixel,
    rebin_image, apply_reference)
from txm.edges import Edge, k_edges
from txm.importers import import_txm_framesets
from txm.xradia import decode_ssrl_params
# from txm import gtk_viewer
from txm import xanes_frameset
from txm import plotter


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

class TXMGtkViewerTest(unittest.TestCase):
    def test_background_frame(self):
        fs = MockFrameset()
        viewer = gtk_viewer.GtkTxmViewer(frameset=fs,
                                         plotter=plotter.DummyGtkPlotter(frameset=fs))
        print(viewer)


if __name__ == '__main__':
    unittest.main()
