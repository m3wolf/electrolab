# -*- coding: utf-8 -*-
#
# Copyright © 2017 Mark Wolfman
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

# flake8: noqa


from unittest import TestCase, mock
import os
import shutil

import numpy as np
import h5py

from scimap.standards import Corundum
from scimap import nca, lmo, exceptions, XRDMap

TESTDIR = os.path.join(os.path.dirname(__file__), "test-data-xrd")
GADDS_HDFFILE = os.path.join(TESTDIR, "xrd-map-gadds.h5")
GADDS_SAMPLE = "xrd-map-gadds"

hdf_34IDE = os.path.join(
    TESTDIR,
    'xrd-map-34-ID-E.h5'
)
group_34IDE = 'xrd-map-34-ID-E'


class XRDMapTest(TestCase):
    def setUp(self):
        self.test_map = XRDMap(Phases=[Corundum],
                               sample_name=group_34IDE,
                               hdf_filename=hdf_34IDE)
    
    def test_set_phases(self):
        """Verify that phases are passed to all the necessary composited objects."""
        new_map = XRDMap(Phases=[nca.NCA], hdf_filename=hdf_34IDE,
                         sample_name=group_34IDE)
        self.assertTrue(
            isinstance(new_map.Phases[0](), nca.NCA)
        )
    
    def test_metric(self):
        new_map = XRDMap(Phases=[nca.NCA], hdf_filename=hdf_34IDE,
                         sample_name=group_34IDE)
        fractions = new_map.metric('phase_fraction')


class MapRefinementTest(TestCase):
    def setUp(self):
        # Create a temporary copy of the hdf data file for testing
        self.h5file = os.path.join(TESTDIR, "xrd-map-gadds-temp.h5")
        if os.path.exists(self.h5file):
            os.remove(self.h5file)
        shutil.copy(GADDS_HDFFILE, self.h5file)
    
    def tearDown(self):
        if os.path.exists(self.h5file):
            os.remove(self.h5file)
    
    def xrdmap(self):
        return XRDMap(sample_name=GADDS_SAMPLE,
                      hdf_filename=self.h5file,
                      Phases=[lmo.MidVPhase, lmo.HighVPhase])
    
    def test_refine_bad_backend(self):
        xrdmap = self.xrdmap()
        with self.assertRaises(exceptions.RefinementError):
            xrdmap.refine_mapping_data(backend='hello')
    
    def test_refine_mapping_data(self):
        xrdmap = self.xrdmap()
        # Do the refinement with mocked data
        DummyRefinement = mock.MagicMock
        DummyRefinement.goodness_of_fit = mock.MagicMock(return_value=np.random.rand(6))
        xrdmap.refine_mapping_data(backend=DummyRefinement)
        # Check that the new groups have been created
        with h5py.File(self.h5file) as f:
            group = f['xrd-map-gadds'] 
            self.assertIn('backgrounds', group.keys())
            self.assertIn('cell_parameters', group.keys())
            self.assertIn('goodness_of_fit', group.keys())
            self.assertIn('phase_fractions', group.keys())
