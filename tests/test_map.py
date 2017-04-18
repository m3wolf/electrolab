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
    os.path.dirname(__file__),
    'test-data-xrd/xrd-map-34-ID-E.hdf'
)

group_34IDE = 'xrd-map-34-ID-E'

class XRDMapTest(unittest.TestCase):
    def setUp(self):
        self.test_map = XRDMap(Phases=[Corundum],
                               sample_name=group_34IDE,
                               hdf_filename=hdf_34IDE)
        self.savefile = 'test-sample.map'

    def tearDown(self):
        try:
            os.remove(self.savefile)
        except FileNotFoundError:
            pass

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
        new_map.metric('phase_ratio')

    # def test_pass_filename(self):
    #     self.assertEqual(
    #         self.test_map.loci[0].filename,
    #         self.test_map.loci[0].xrdscan.filename
    #     )

    # def test_save_map(self):
    #     # Set some new data
    #     self.test_map.diameter = 5
    #     self.test_map.coverage = 0.33
    #     self.test_map.save()
    #     # Make sure savefile was created
    #     self.assertTrue(
    #         os.path.isfile(self.savefile)
    #     )
    #     # Load from file
    #     new_map = XRDMap()
    #     new_map.load(filename=self.savefile)
    #     self.assertEqual(new_map.diameter, self.test_map.diameter)
    #     self.assertEqual(new_map.coverage, self.test_map.coverage)

    # def test_save_loci(self):
    #     """Does the save routine properly save the loci list."""
    #     original_locus = self.test_map.loci[0]
    #     original_locus.metric = 200
    #     original_locus.filebase = 'nonsense'
    #     original_locus.cube_coords = Cube(20, 20, 20)
    #     original_locus.diffractogram = 'Gibberish'
    #     self.test_map.save()
    #     new_map = XRDMap()
    #     new_map.load(self.savefile)
    #     new_locus = new_map.loci[0]
    #     self.assertEqual(new_locus.metric, original_locus.metric)
    #     self.assertEqual(new_locus.filebase, original_locus.filebase)
    #     self.assertEqual(new_locus.cube_coords, original_locus.cube_coords)
    #     self.assertEqual(new_locus.diffractogram, original_locus.diffractogram)

    # def test_save_refinement(self):
    #     original = self.test_map.loci[0].refinement
    #     original.spline = (1, 3, 5)
    #     self.test_map.save()
    #     new_map = XRDMap()
    #     new_map.load(self.savefile)
    #     new_refinement = new_map.loci[0].refinement
    #     self.assertEqual(new_refinement.spline, original.spline)

    # def test_save_phases(self):
    #     original = self.test_map.loci[0].phases[0]
    #     original.scale_factor = 100
    #     self.test_map.save()
    #     new_map = XRDMap(phases=[Corundum])
    #     new_map.load(self.savefile)
    #     new_phase = new_map.loci[0].phases[0]
    #     self.assertEqual(new_phase.scale_factor, original.scale_factor)

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

    def test_refine_phase_ratio(self):
        xrdmap = self.xrdmap()
        # Check that the right groups are not created before refining
        with h5py.File(self.h5file) as f:
            group = f['xrd-map-gadds']
            self.assertNotIn('background', group.keys())
            self.assertNotIn('cell_parameters', group.keys())
            self.assertNotIn('goodness_of_fit', group.keys())
            self.assertNotIn('phase_fractions', group.keys())
        # Do the actual refinement
        xrdmap.refine_mapping_data()
        # Check that the new groups have been created
        with h5py.File(self.h5file) as f:
            group = f['xrd-map-gadds']
            self.assertIn('backgrounds', group.keys())
            self.assertIn('cell_parameters', group.keys())
            self.assertIn('goodness_of_fit', group.keys())
            self.assertIn('phase_fractions', group.keys())
