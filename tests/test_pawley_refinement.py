# -*- coding: utf-8 -*-
#
# Copyright © 2018 Mark Wolf
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
import warnings
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebval

from scimap import XRDScan, standards
from scimap.utilities import q_to_twotheta, twotheta_to_q
from scimap.peakfitting import remove_peak_from_df
from scimap.pawley_refinement import PawleyRefinement, predict_diffractogram, gaussian, cauchy

TESTDIR = os.path.join(os.path.dirname(__file__), "test-data-xrd")
COR_BRML = os.path.join(TESTDIR, 'corundum.brml')

@unittest.skip
class PawleyRefinementTest(unittest.TestCase):
    wavelengths = (
            (1.5406, 1), # Kα1
            (1.5444, 0.5), # Kα2
        )
    
    def test_refine(self):
        # Prepare the test objects
        scan = XRDScan(COR_BRML, phase=standards.Corundum())
        two_theta = q_to_twotheta(scan.scattering_lengths, wavelength=scan.wavelength)
        refinement = PawleyRefinement(wavelengths=self.wavelengths,
                                      phases=[scan.phases[0]],
                                      num_bg_coeffs=9)
        
        # Set starting parameters a bit closer to keep from taking too long
        scan.phases[0].unit_cell.a = 4.755
        scan.phases[0].unit_cell.c = 12.990
        for r in scan.phases[0].reflection_list:
            r.intensity *= 200
        # Do the refinement
        refinement.refine(two_theta=two_theta, intensities=scan.intensities)
        predicted = refinement.predict(two_theta=two_theta)
        actual = scan.diffractogram.counts.values
        refinement.plot(two_theta=two_theta, intensities=scan.intensities)
        plt.show()
        # Check scale factors
        scale = refinement.scale_factor(two_theta=two_theta)
        print(scale)
        # Check phase fractions
        fracs = refinement.phase_fractions(two_theta=two_theta)
        np.testing.assert_almost_equal(fracs, (1, ))
        # Check background fitting
        bg = refinement.background(two_theta=two_theta)
        self.assertEqual(bg.shape, actual.shape)
        # Check refined unit-cell parameters
        unit_cell = refinement.unit_cells()
        self.assertAlmostEqual(
            unit_cell[0],
            4.758877,
            places=3,
        )
        self.assertAlmostEqual(
            unit_cell[2],
            12.992877,
            places=3
        )
        peak_widths = refinement.peak_breadths(two_theta, scan.intensities)
        self.assertAlmostEqual(peak_widths[0], 0.046434, places=5)        
        # Check that the refinement is of sufficient quality
        error = np.sqrt(np.sum((actual-predicted)**2))/len(actual)
        self.assertLess(error, 1.5)
    
    def test_reflection_list(self):
        corundum = standards.Corundum()
        # Determine the min and max values for 2θ range
        two_theta_range = (10, 80)
        d_min = 1.5406 / 2 / np.sin(np.radians(40))
        d_max = 1.5444 / 2 / np.sin(np.radians(5))
        # Calculate expected reflections
        d_list = [corundum.unit_cell.d_spacing(r.hkl) for r in corundum.reflection_list]
        d_list = np.array(d_list)
        h_list = np.array([r.intensity for r in corundum.reflection_list])
        size_list = np.array([corundum.crystallite_size for d in d_list])
        strain_list = np.array([corundum.strain for d in d_list])
        # Restrict values to only those valid for this two-theta range
        valid_refs = np.logical_and(d_list >= d_min, d_list <= d_max)
        d_list = d_list[valid_refs]
        h_list = h_list[valid_refs]
        size_list = size_list[valid_refs]
        strain_list = strain_list[valid_refs]
        # Calculate the actual list and compare
        ref_list = tuple(zip(d_list, h_list, size_list, strain_list))
        refinement = PawleyRefinement(phases=[corundum],
                                      wavelengths=self.wavelengths)
        output = refinement.reflection_list(two_theta=two_theta_range, clip=True)
        np.testing.assert_equal(output, ref_list)
    
    def test_predict_diffractogram(self):
        two_theta = np.linspace(10, 80, num=40001)
        wavelengths = self.wavelengths
        bg_coeffs = np.ones(shape=(10,))
        bg_x = np.linspace(10/90, 80/90, num=len(two_theta)) - 1
        bg = chebval(bg_x, bg_coeffs)
        # Check background
        result = predict_diffractogram(
            two_theta=two_theta,
            wavelengths=wavelengths,
            background_coeffs=bg_coeffs,
            reflections=(),
        )
        np.testing.assert_almost_equal(result, bg)
        # Check list of reflections
        reflections = np.array((
            (3.0, 79, 100, 0),
            (3/1.41, 15, 100, 0),
            (1.4, 41, 100, 0),
            (0.5, 27, 100, 0),
        ))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = predict_diffractogram(
                two_theta=two_theta,
                wavelengths=wavelengths,
                background_coeffs=(0,),
                u=0.01, v=0, w=0,
                reflections=reflections,
                lg_mix=0.5,
            )
            # Check that a warning is raised for the invalid
            # reflection (0.5Å)
            self.assertEqual(len(w), 2, "No warning raised")
        # plt.plot(two_theta, result)
        # plt.show()
        self.assertAlmostEqual(np.max(result), 79, places=0)
    
    def test_peak_shapes(self):
        x = np.linspace(0, 10, num=501)
        beta = 0.05
        height = 20
        gauss = gaussian(x, center=5, height=height, breadth=beta)
        self.assertEqual(np.max(gauss), height)
        area = np.trapz(gauss, x=x)
        self.assertAlmostEqual(area/height, beta)
        # Now check Cauchy function
        cauch = cauchy(x, center=5, height=height, breadth=beta)
        self.assertEqual(np.max(cauch), height)
        # (Cauchy beta is less reliable because the tails have more weight)
        area = np.trapz(cauch, x=x)
        self.assertAlmostEqual(area/height, beta, places=2)
        # plt.plot(x, gauss)
        # plt.plot(x, cauch)
        # plt.show()
