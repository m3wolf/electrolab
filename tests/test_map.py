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

# flake8: noqa


import unittest
import os
import shutil
import h5py

from scimap import XRDMap
from scimap.standards import Corundum
from scimap import nca, lmo

TESTDIR = os.path.join(os.path.dirname(__file__), "test-data-xrd")
GADDS_HDFFILE = os.path.join(TESTDIR, "xrd-map-gadds.h5")
GADDS_SAMPLE = "xrd-map-gadds"

hdf_34IDE = os.path.join(
    TESTDIR,
    'xrd-map-34-ID-E.h5'
)
group_34IDE = 'xrd-map-34-ID-E'

class XRDMapTest(unittest.TestCase):
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


class MapRefinementTest(unittest.TestCase):
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
    
    @unittest.expectedFailure
    def test_refine_mapping_data(self):
        xrdmap = self.xrdmap()
        # Check that the right groups are not created before refining
        with h5py.File(self.h5file) as f:
            group = f['xrd-map-gadds']
            # Get rid of exisiting data groups
            groups = ('background', 'cell_parameters', 'goodness_of_fit', 'phase_fractions')
            for del_grp in groups:
                if del_grp in group.keys():
                    del group[del_grp]
                self.assertNotIn(del_grp, group.keys())
            # del group['cell_parameters']
            # self.assertNotIn('cell_parameters', group.keys())
            # del group['goodness_of_fit']
            # self.assertNotIn('goodness_of_fit', group.keys())
            # # del group['phase_fractions']
            # self.assertNotIn('phase_fractions', group.keys())
        # Do the actual refinement
        xrdmap.refine_mapping_data(backend="pawley")
        # Check that the new groups have been created
        with h5py.File(self.h5file) as f:
            group = f['xrd-map-gadds'] 
            self.assertIn('backgrounds', group.keys())
            self.assertIn('cell_parameters', group.keys())
            self.assertIn('goodness_of_fit', group.keys())
            self.assertIn('phase_fractions', group.keys())
