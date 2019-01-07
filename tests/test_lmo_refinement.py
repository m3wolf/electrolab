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

"""Tests for refining LiM2O4 cathodes."""

import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from scimap import XRDScan, standards, lmo, tubes
from scimap.peakfitting import remove_peak_from_df
from scimap.native_refinement import NativeRefinement, contains_peak, peak_area

TESTDIR = os.path.join(os.path.dirname(__file__), "test-data-xrd")
COR_BRML = os.path.join(TESTDIR, 'corundum.brml')


class LMOTwoPhaseMapTest(unittest.TestCase):
    """Tests for the two-phase (High-SoC) regime."""
    plt_file = os.path.join(TESTDIR, 'LMO-two-phase.plt')
    def setUp(self):
        self.scan = XRDScan(self.plt_file)
        wavelengths = tubes['Cu'].wavelengths
        self.refinement = lmo.TwoPhaseRefinement(wavelengths=wavelengths)
    
    def test_background(self):
        TTs = self.scan.two_theta
        Is = self.scan.intensities
        bg = self.refinement.background(TTs, Is)
        # Check that background has the right shape
        self.assertEqual(bg.shape, TTs.shape)
        # Check that the background doesn't go past 7, which is about
        # as high as the noise. Higher values mean it's fitting peaks
        self.assertLess(np.max(bg), 7)
    
    def test_fit_peaks(self):
        TTs = self.scan.two_theta
        Is = self.scan.intensities
        peaks = self.refinement.fit_peaks(TTs, Is)
        # Check that the number of peaks are fit correctly
        self.assertEqual(len(peaks), 6)
    
    def test_overall_fit(self):
        TTs = self.scan.two_theta
        Is = self.scan.intensities
        # Check that the residuals of the fit are low
        predicted = self.refinement.predict(TTs, Is)
        rms_error = np.sqrt(np.mean((Is-predicted)**2))
        self.assertLess(rms_error, 0.55)
    
    def test_phase_fractions(self):
        TTs = self.scan.two_theta
        Is = self.scan.intensities
        fractions = self.refinement.phase_fractions(TTs, Is)
        np.testing.assert_almost_equal(fractions, [0.555, 0.445], decimal=3)
    
    def test_goodness_of_fit(self):
        TTs = self.scan.two_theta
        Is = self.scan.intensities
        goodness = self.refinement.goodness_of_fit(TTs, Is)
        self.assertGreater(goodness, 0.85)
    
    def test_cell_params(self):
        TTs = self.scan.two_theta
        Is = self.scan.intensities
        cell_params = self.refinement.cell_params(TTs, Is)
        expected = [(8.146, 8.146, 8.146, 90, 90, 90),
                    (8.065, 8.065, 8.065, 90, 90, 90)]
        np.testing.assert_almost_equal(cell_params, expected, decimal=3)
    
    def test_scale_factor(self):
        TTs = self.scan.two_theta
        Is = self.scan.intensities
        scale_factor = self.refinement.scale_factor(TTs, Is)
        self.assertAlmostEqual(scale_factor, 47.8, places=1)
    
    def test_broadenings(self):
        TTs = self.scan.two_theta
        Is = self.scan.intensities
        broadenings = self.refinement.broadenings(TTs, Is)
        self.assertEqual(len(broadenings), 2)
        np.testing.assert_almost_equal(
            broadenings, [1.19, 1.26], decimal=2)
    
    def test_smooth_data(self):
        TTs = self.scan.two_theta
        Is = self.scan.intensities
        smoothed = self.refinement.smooth_data(Is)
        self.assertEqual(Is.shape, smoothed.shape)
    
    def test_bad_refinement(self):
        """This .plt file did not refine originally, so let's fix it."""
        plt_file = os.path.join(TESTDIR, '1C_charged_plateau-map-129.plt')
        scan = XRDScan(plt_file)
        TTs = scan.two_theta
        Is = scan.intensities
        predicted = self.refinement.predict(TTs, Is)
        bg = self.refinement.background(TTs, Is)
        fractions = self.refinement.phase_fractions(TTs, Is)
        unit_cells = self.refinement.cell_params(TTs, Is)
        np.testing.assert_almost_equal(fractions, [0.897, 0.103], decimal=3)
    
    def test_bad_phase_fraction(self):
        """This .plt gaves a wrong value when refining phase fraction."""
        plt_file = os.path.join(TESTDIR, '1C_charged_plateau-map-13c.plt')
        scan = XRDScan(plt_file)
        TTs = scan.two_theta
        Is = scan.intensities
        predicted = self.refinement.predict(TTs, Is)
        bg = self.refinement.background(TTs, Is)
        fractions = self.refinement.phase_fractions(TTs, Is)
        np.testing.assert_almost_equal(fractions, [0.87, 0.13], decimal=2)


class LMOSolutionMapTest(unittest.TestCase):
    """Tests for the solid-solution (Low SoC) regime."""
    plt_file = os.path.join(TESTDIR, 'LMO-solid-solution.plt')
    def setUp(self):
        self.scan = XRDScan(self.plt_file)
        wavelengths = tubes['Cu'].wavelengths
        self.refinement = lmo.SolidSolutionRefinement(wavelengths=wavelengths)
    
    def test_background(self):
        TTs = self.scan.two_theta
        Is = self.scan.intensities
        bg = self.refinement.background(TTs, Is)
        # Check that background has the right shape
        self.assertEqual(bg.shape, TTs.shape)
        # Check that the background doesn't go past 7, which is about
        # as high as the noise. Higher values mean it's fitting peaks
        self.assertLess(np.max(bg), 7)
    
    def test_phase_fractions(self):
        TTs = self.scan.two_theta
        Is = self.scan.intensities
        # Check that phase ratios are correct
        ratios = self.refinement.phase_fractions(TTs, Is)
        np.testing.assert_equal(ratios, [1.])
    
    def test_fit_peaks(self):
        TTs = self.scan.two_theta
        Is = self.scan.intensities
        peaks = self.refinement.fit_peaks(TTs, Is)
        # Check that the number of peaks are fit correctly
        self.assertEqual(len(peaks), 3)
    
    def test_overall_fit(self):
        TTs = self.scan.two_theta
        Is = self.scan.intensities
        # Check that the residuals of the fit are low
        predicted = self.refinement.predict(TTs, Is)
        rms_error = np.sqrt(np.mean((Is-predicted)**2))
        self.assertLess(rms_error, 0.65)
    
    def test_goodness_of_fit(self):
        TTs = self.scan.two_theta
        Is = self.scan.intensities
        goodness = self.refinement.goodness_of_fit(TTs, Is)
        self.assertGreater(goodness, 0.84)
    
    def test_cell_params(self):
        TTs = self.scan.two_theta
        Is = self.scan.intensities
        cell_params = self.refinement.cell_params(TTs, Is)
        expected = [(8.168, 8.168, 8.168, 90, 90, 90)]
        np.testing.assert_almost_equal(cell_params, expected, decimal=3)
    
    def test_scale_factor(self):
        TTs = self.scan.two_theta
        Is = self.scan.intensities
        scale_factor = self.refinement.scale_factor(TTs, Is)
        self.assertAlmostEqual(scale_factor, 42.3, places=1)
    
    def test_broadenings(self):
        TTs = self.scan.two_theta
        Is = self.scan.intensities
        broadenings = self.refinement.broadenings(TTs, Is)
        self.assertEqual(len(broadenings), 1)
        np.testing.assert_almost_equal(
            broadenings, [1.110], decimal=3)
    
    def test_bad_refinement(self):
        # This specific plot did not fit well
        plt_file = os.path.join(TESTDIR, 'charged_2C_to47V_quarterlithium-map-c0.plt')
        print(plt_file)
        scan = XRDScan(plt_file)
        TTs = scan.two_theta
        Is = scan.intensities
        smoothed = self.refinement.smooth_data(Is)
        # plt.plot(TTs, Is)
        # plt.plot(TTs, self.refinement.predict(TTs, Is))
        # plt.plot(TTs, smoothed)
        # plt.show()


class LMOQuarterLithiumTest(unittest.TestCase):
    """Tests for an analysis involving the full range of cell parameters.
    
    Data files are from a 2C charge where the lithium anode is half
    the diameter of the cathode.
    
    - LMO-quarterlithium-charged.plt -> LMO-NEI/NEI-Pg-S10-10-pre-soaking-frames/map-0.plt
    - LMO-quarterlithium-discharged.plt -> LMO-NEI/NEI-Pg-S10-10-pre-soaking-frames/map-b1.plt
    - LMO-quarterlithium-discharged-2.plt -> LMO-NEI/NEI-Pg-S10-10-pre-soaking-frames/map-c9.plt
    - LMO-quarterlithium-discharged-3.plt -> LMO-NEI/NEI-Pg-S10-10-pre-soaking-frames/map-102.plt
    
    """
    charged_file = os.path.join(TESTDIR, 'LMO-quarterlithium-charged.plt')
    discharged_file = os.path.join(TESTDIR, 'LMO-quarterlithium-discharged.plt')
    discharged_file2 = os.path.join(TESTDIR, 'LMO-quarterlithium-discharged-2.plt')
    discharged_file3 = os.path.join(TESTDIR, 'LMO-quarterlithium-discharged-3.plt')
    def setUp(self):
        wavelengths = tubes['Cu'].wavelengths
        self.refinement = lmo.LmoFullRefinement(wavelengths=wavelengths)
    
    def test_background_charged(self):
        scan = XRDScan(self.charged_file)
        TTs = scan.two_theta
        Is = scan.intensities
        bg = self.refinement.background(TTs, Is)
        # Check that background has the right shape
        self.assertEqual(bg.shape, TTs.shape)
        # Check that the background doesn't go past 7, which is about
        # as high as the noise. Higher values mean it's fitting peaks
        self.assertLess(np.max(bg), 7)
    
    def test_background_discharged(self):
        scan = XRDScan(self.discharged_file)
        TTs = scan.two_theta
        Is = scan.intensities
        bg = self.refinement.background(TTs, Is)
        # Check that background has the right shape
        self.assertEqual(bg.shape, TTs.shape)
        # Check that the background doesn't go past 7, which is about
        # as high as the noise. Higher values mean it's fitting peaks
        self.assertLess(np.max(bg), 7)
    
    def test_overall_fit_discharged2(self):
        scan = XRDScan(self.discharged_file2)
        TTs = scan.two_theta
        Is = scan.intensities
        # Check that the residuals of the fit are low
        predicted = self.refinement.predict(TTs, Is)
        rms_error = np.sqrt(np.mean((Is-predicted)**2))
        self.assertLess(rms_error, 0.65)
    
    def test_overall_fit_discharged3(self):
        scan = XRDScan(self.discharged_file3)
        TTs = scan.two_theta
        Is = scan.intensities
        # Check that the residuals of the fit are low
        predicted = self.refinement.predict(TTs, Is)
        rms_error = np.sqrt(np.mean((Is-predicted)**2))
        # plt.plot(TTs, Is)
        # plt.plot(TTs, predicted)
        # plt.show()
        self.assertLess(rms_error, 0.65)        
    
    def test_overall_fit_charged(self):
        scan = XRDScan(self.charged_file)
        TTs = scan.two_theta
        Is = scan.intensities
        # Check that the residuals of the fit are low
        predicted = self.refinement.predict(TTs, Is)
        rms_error = np.sqrt(np.mean((Is-predicted)**2))
        self.assertLess(rms_error, 0.65)
    
    def test_overall_fit_discharged(self):
        scan = XRDScan(self.discharged_file)
        TTs = scan.two_theta
        Is = scan.intensities
        # Check that the residuals of the fit are low
        predicted = self.refinement.predict(TTs, Is)
        rms_error = np.sqrt(np.mean((Is-predicted)**2))
        # self.assertLess(rms_error, 0.65)
    
    def test_cell_params_charged(self):
        scan = XRDScan(self.charged_file)
        TTs = scan.two_theta
        Is = scan.intensities
        cell_params = self.refinement.cell_params(TTs, Is)
        expected = [[ 8.141,  8.141,  8.141, 90.   , 90.   , 90.   ],
                    [ 8.061,  8.061,  8.061, 90.   , 90.   , 90.   ]]
        np.testing.assert_almost_equal(cell_params, expected, decimal=3)        
    
    def test_cell_params_discharged(self):
        scan = XRDScan(self.discharged_file)
        TTs = scan.two_theta
        Is = scan.intensities
        cell_params = self.refinement.cell_params(TTs, Is)
        expected = [8.233, 8.233, 8.233, 90, 90, 90]
        # Only check for the first phase, the second phase is nothing
        np.testing.assert_almost_equal(cell_params[0], expected, decimal=3)        
