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

from scimap.fullprof_refinement import FullProfPhase, FullprofRefinement as FullprofProfileMatch
from scimap.lmo import CubicLMO
from scimap.unitcell import UnitCell, CubicUnitCell, HexagonalUnitCell
from scimap.scan import XRDScan
from scimap.standards import Corundum, Aluminum


class FullProfProfileTest(unittest.TestCase):
    def setUp(self):
        # Base parameters determine from manual refinement
        class FPCorundum(Corundum):
            unit_cell = HexagonalUnitCell(a=4.758637, c=12.991814)
            u = 0.00767
            v = -0.003524
            w = 0.002903
            x = 0.001124
            eta = 0.511090
            isotropic_temp = 33.314
        self.scan = XRDScan('test-sample-frames/corundum.brml',
                               phase=FPCorundum())
        self.refinement = FullprofProfileMatch(scan=self.scan)
        self.refinement.zero = -0.003820
        self.refinement.displacement = 0.0012
        self.refinement.bg_coeffs = [129.92, -105.82, 108.32, 151.85, -277.55, 91.911]
        # self.refinement.keep_temp_files = True
    
    @unittest.expectedFailure
    def test_jinja_context(self):
        context = self.refinement.pcrfile_context()
        self.assertEqual(len(context['phases']), 1)
        phase1 = context['phases'][0]
        self.assertEqual(
            phase1['spacegroup'],
            'R -3 C'
        )
        self.assertEqual(
            phase1['vals']['a'],
            4.758637
        )
        self.assertEqual(
            phase1['vals']['u'],
            self.scan.phases[0].u
        )
        self.assertEqual(
            context['bg_coeffs'],
            self.refinement.bg_coeffs
        )
        self.assertEqual(
            context['displacement_codeword'],
            0
        )
        self.assertEqual(
            context['phases'][0]['vals']['I_g'],
            0
        )
    
    @unittest.expectedFailure
    def test_refine_background(self):
        # Set bg coeffs to something wrong
        self.refinement.bg_coeffs = [0, 0, 0, 0, 0, 0]
        self.refinement.refine_background()
        self.assertTrue(
            self.refinement.is_refined['background']
        )
        # Based on manual refinement in fullprof (winplotr-2006)
        self.assertTrue(
            0 < self.refinement.chi_squared < 10,
            'Χ² is too high: {}'.format(self.refinement.chi_squared)
        )
        self.assertAlmostEqual(
            self.refinement.bg_coeffs,
            [132.87, -35.040, -5.58, 0, 0, 0]
        )
    
    @unittest.expectedFailure
    def test_refine_displacement(self):
        # Set sample displacement to something wrong
        self.refinement.displacement = 0
        self.refinement.refine_displacement()
        self.assertTrue(
            0 < self.refinement.chi_squared < 10,
            'Χ² is too high: {}'.format(self.refinement.chi_squared)
        )
        # Based on manual refinement in fullprof
        self.assertAlmostEqual(
            self.refinement.displacement,
            0.0054
        )
    
    @unittest.expectedFailure
    def test_refine_unit_cell(self):
        # Set unit cell parameters off by a little bit
        phase = self.scan.phases[0]
        phase.unit_cell.a = 4.75
        phase.unit_cell.c = 12.982
        self.refinement.refine_unit_cells()
        self.assertTrue(self.refinement.is_refined['unit_cells'])
        self.assertTrue(self.refinement.chi_squared < 10)
        self.assertAlmostEqual(phase.unit_cell.a, 4.758637,
                                      tolerance=0.001)
        self.assertAlmostEqual(phase.unit_cell.c, 12.991814,
                                      tolerance=0.001)


class FullProfLmoTest(unittest.TestCase):
    """Check refinement using data from LiMn2O4 ("NEI")"""
    def setUp(self):
        # Base parameters determine from manual refinement
        class LMOHighV(FullProfPhase, CubicLMO):
            unit_cell = CubicUnitCell(a=8.052577)
            isotropic_temp = 0.19019
            u = -0.000166
            v = 0.120548
            w = 0.003580
            I_g = 0.000142
            eta = 0.206420
            x = 0.007408
        class LMOMidV(FullProfPhase, CubicLMO):
            unit_cell = CubicUnitCell(a=8.122771)
            isotropic_temp = -0.45434
            u = 0.631556
            v = -0.115778
            w = 0.019247
            I_g = -0.000539
            eta = 0.923930
            x = -0.006729
        self.scan = XRDScan('test-sample-frames/lmo-two-phase.brml',
                               phases=[LMOHighV(), LMOMidV()])
        self.refinement = FullprofProfileMatch(scan=self.scan)
        # Base parameters determined by manual refinement
        self.refinement.bg_coeffs = [71.297, -50.002, 148.13, -150.13, -249.84, 297.01]
        self.refinement.zero = 0.044580
        self.refinement.displacement = 0.000320
        self.refinement.transparency = -0.00810
    
    @unittest.expectedFailure
    def test_scale_factors(self):
        self.refinement.refine_scale_factors()
        self.assertTrue(
            self.refinement.is_refined['scale_factors']
        )
        self.assertTrue(
            self.refinement.chi_squared < 10
        )
        self.assertAlmostEqual(
            self.scan.phases[0].scale_factor,
            37.621
        )
        self.assertAlmostEqual(
            self.scan.phases[1].scale_factor,
            40.592
        )
