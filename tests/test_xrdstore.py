# -*- coding: utf-8 -*-
#
# Copyright © 2017 Mark Wolf
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
from unittest import mock
import os

import numpy as np
import h5py

from scimap.xrdstore import XRDStore, StoreDescriptor
from cases import HDFTestCase

TESTDIR = os.path.dirname(__file__)
# GADDS_HDFFILE = os.path.join(TESTDIR, 'xrd-map-gadds.h5')


class StoreDescriptorTests(unittest.TestCase):
    class TestStore():
        replace_dataset = mock.MagicMock()
        get_dataset = mock.MagicMock()
        my_attr = StoreDescriptor('my_attr', dtype='float32')
    
    def test_setter(self):
        store = self.TestStore()
        test_data = np.random.rand(4, 4)
        store.my_attr = test_data
        # Check that the replace_dataset method was called
        self.assertEqual(store.replace_dataset.call_count, 1)
        set_attr_name = store.replace_dataset.call_args[1]['name']
        self.assertEqual(set_attr_name, 'my_attr')
        set_value = store.replace_dataset.call_args[1]['data']
        np.testing.assert_equal(set_value, test_data)
        set_dtype = store.replace_dataset.call_args[1]['dtype']
        self.assertEqual(set_dtype, 'float32')
    
    def test_getter(self):
        # Set up the test data
        store = self.TestStore()
        test_data = np.random.rand(4, 4)
        store.get_dataset.return_value = test_data
        # Retrieve the data and check that the right methods were called
        self.assertEqual(store.get_dataset.call_count, 0)
        new_data = store.my_attr
        self.assertEqual(store.get_dataset.call_count, 1)
        np.testing.assert_array_equal(new_data, test_data)


class XRDStoreTests(unittest.TestCase):
    temp_hdffile = os.path.join(TESTDIR, 'test-data-xrd/temp-file.h5')
    
    def tearDown(self):
        # Remove any temporary HDF files
        if os.path.exists(self.temp_hdffile):
            os.remove(self.temp_hdffile)
    
    def test_get_dataset(self):
        hdf_filename = self.temp_hdffile
        groupname = 'xrd-map-gadds'
        test_data = np.random.rand(16, 16)
        # Create a temporary HDF5 file
        with h5py.File(hdf_filename) as h5file:
            h5file.create_group(groupname)
            h5file[groupname].create_dataset('test-ds', data=test_data)
        # Load the store and create a group
        store = XRDStore(hdf_filename=hdf_filename, groupname=groupname)
        h5file = h5py.File(hdf_filename, mode='r')
        with store, h5file:
            new_ds = store.get_dataset('test-ds')
            np.testing.assert_equal(new_ds, test_data)
    
    def test_replace_dataset(self):
        hdf_filename = self.temp_hdffile
        groupname = 'xrd-map-gadds'
        test_data = np.random.rand(16, 16)
        # Create a temporary HDF5 file
        with h5py.File(hdf_filename) as h5file:
            h5file.create_group(groupname)
        # Load the store and create a group
        store = XRDStore(hdf_filename=hdf_filename, groupname=groupname, mode='r+')
        h5file = h5py.File(hdf_filename, mode='r')
        with store, h5file:
            h5grp = h5file[groupname]
            store.replace_dataset('test-ds', test_data)
            new_ds = h5grp['test-ds']
            np.testing.assert_equal(new_ds, test_data)
            # Now check if it replaces an exisiting dataset
            test_data = np.random.rand(12, 12)
            store.replace_dataset('test-ds', test_data)
            new_ds = h5grp['test-ds']
            np.testing.assert_equal(new_ds, test_data)
            # Check if attributes are preserved
            new_ds.attrs['my_attr'] = 'my value'
            store.replace_dataset('test-ds', test_data)
            new_ds = h5grp['test-ds']
            self.assertEqual(new_ds.attrs['my_attr'], 'my value')
            # Check if dtype is setUp
            store.replace_dataset('test-ds', test_data, dtype='float32')
            new_ds = h5grp['test-ds']
            self.assertEqual(new_ds.dtype, np.dtype('float32'))
    
    def test_descriptors(self):
        """Open a XRD store and check getting all the descriptors.
        
        Check the descriptors with the actual HDF5 data.
        
        """
        hdf_filename = os.path.join(TESTDIR, 'test-data-xrd/test_map_gadds.h5')
        h5file = h5py.File(hdf_filename, mode='r')
        store = XRDStore(hdf_filename=hdf_filename, groupname='xrd-map-gadds')
        with store, h5file:
            h5grp = h5file['xrd-map-gadds']
            np.testing.assert_equal(store.positions, h5grp['positions'])
            # h5grp.create_dataset('intensities', data=np.random.rand(397, 20))
            np.testing.assert_equal(store.goodness_of_fit, h5grp['goodness_of_fit'])
            np.testing.assert_equal(store.intensities, h5grp['intensities'])
        
    # def test_step_size(self):
    #     store = XRDStore(hdf_filename=self.hdf_filename,
    #                      groupname="xrd-map-gadds")
    #     # For now, just check that the value can be retrieved without
    #     # throwing an error
    #     step_size = store.step_size
    
    # def test_step_unit(self):
    #     store = XRDStore(hdf_filename=self.hdf_filename,
    #                      groupname="xrd-map-gadds")
    #     store.step_unit = 'um'
    #     step_unit = store.step_unit
    #     self.assertEqual(step_unit, 'um')
    
    # def test_scale_factor(self):
    #     store = XRDStore(hdf_filename=self.hdf_filename,
    #                      groupname="xrd-map-gadds")
    #     store.scale_factor
