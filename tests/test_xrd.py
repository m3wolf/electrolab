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

# flake8: noqa

import unittest
import math
import os
import shutil
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

if __name__ == '__main__':
    # Set backend so matplotlib doesn't try and show plots
    import matplotlib
    matplotlib.use('Agg')
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import h5py

from scimap import exceptions, gadds, prog, units
from scimap.peakfitting import PeakFit, remove_peak_from_df, discrete_fwhm
from scimap import lmo, nca
from scimap.nca import NCA
from scimap.lmo import CubicLMO
from scimap.unitcell import UnitCell, CubicUnitCell, HexagonalUnitCell
from scimap.reflection import Reflection, hkl_to_tuple
from scimap.scan import XRDScan
from scimap.coordinates import Cube
from scimap.standards import Corundum, Aluminum
from scimap.importers import import_gadds_map
from scimap.utilities import q_to_twotheta, twotheta_to_q
from scimap.peak import XRDPeak
from scimap.xrd_map import XRDMap
from scimap.native_refinement import NativeRefinement, contains_peak
from scimap.adapters import BrukerRawFile, BrukerBrmlFile, BrukerXyeFile, BrukerPltFile

TESTDIR = os.path.dirname(__file__)
GADDS_SAMPLE = "xrd-map-gadds-temp"


prog.quiet = True


corundum_path = os.path.join(
    os.path.dirname(__file__),
    'test-data-xrd/corundum.brml'
)

hdf_34IDE = os.path.join(
    os.path.dirname(__file__),
    'test-data-xrd/xrd-map-34-ID-E.h5'
)

group_34IDE = 'xrd-map-34-ID-E'


# Some phase definitions for testing
class LMOHighV(lmo.CubicLMO):
    unit_cell = CubicUnitCell(a=8.05)
    diagnostic_hkl = '333'
    reflection_list = [
        Reflection('333', (58.5, 59.3))
    ]


class LMOMidV(lmo.CubicLMO):
    unit_cell = CubicUnitCell(a=8.13)
    diagnostic_hkl = '333'
    reflection_list = [
        Reflection('333', (59.3, 59.9))
    ]

class LMOLowAngle(lmo.CubicLMO):
    diagnostic_hkl = '311'


class QTwoThetaTest(unittest.TestCase):
    """Test for proper conversion between scattering length (Q) and 2θ"""
    wavelength = 1.5418
    def test_twotheta_to_q(self):
        # Without units
        q = twotheta_to_q(10, wavelength=self.wavelength)
        self.assertEqual(q, 0.71035890811831559)
        # With numpy array
        q = twotheta_to_q(np.array((10,)), wavelength=self.wavelength)
        self.assertEqual(q[0], 0.71035890811831559)
        # In radians
        q = twotheta_to_q(np.radians(10), wavelength=self.wavelength,
                          degrees=False)
        self.assertEqual(q, 0.71035890811831559)
    
    def test_q_to_twotheta(self):
        # Regular float number
        twotheta = q_to_twotheta(0.71, wavelength=self.wavelength)
        self.assertEqual(twotheta, 9.9949346551020319)
        # With numpy array
        twotheta = q_to_twotheta(np.array((0.71,)), wavelength=self.wavelength)
        self.assertEqual(twotheta[0], 9.9949346551020319)


class PeakTest(unittest.TestCase):
    def test_split_parameters(self):
        peak = XRDPeak()
        # Put in some junk data so it will actually split
        peak.fit_list = ['a', 'b']
        peak.num_peaks = 2
        fullParams = (1, 2, 3, 4, 5, 6)
        splitParams = peak.split_parameters(fullParams)
        self.assertEqual(
            splitParams,
            [(1, 2, 3), (4, 5, 6)]
        )
    
    def gaussian_curve(self, x=None, height=1, center=0, fwhm=1):
        if x is None:
            x = np.linspace(-5, 5, num=500)
        a = height
        b = center
        c = fwhm / (2*math.sqrt(2*math.log(2)))
        y = a * np.exp(-(x-b)**2/(2*c**2))
        return x, y
    
    def test_discrete_fwhm(self):
        """This test checks that the full-width half-max is properly
        approximated."""
        expected_fwhm = 1.
        # Construct a Gaussian curve
        x, y = self.gaussian_curve(x=np.linspace(-5, 5, num=5000), fwhm=expected_fwhm)
        # Calculate FWHM
        fwhm = discrete_fwhm(x, y)
        self.assertAlmostEqual(fwhm, expected_fwhm, places=1)
    
    @unittest.expectedFailure
    def test_initial_parameters(self):
        # Does the class guess reasonable starting values for peak fitting
        peakScan = XRDScan(corundum_path, phase=Corundum())
        df = peakScan.diffractogram[2.38:2.52]
        peak = XRDPeak(method="gaussian")
        guess = peak.guess_parameters(x=df.index, y=df.counts.values)
        # Should be two peaks present
        self.assertEqual(len(guess), 2)
        tolerance = 0.001
        # Returns two peaks for kα₁ and kα₂
        p1, p2 = guess
        self.assertAlmostEqual(p1.height, 426.604, tolerance=tolerance)
        self.assertAlmostEqual(p1.center, 35.123, tolerance=tolerance)
        self.assertAlmostEqual(p1.width, 0.02604, tolerance=tolerance)
        self.assertAlmostEqual(p2.height, 213.302, tolerance=tolerance)
        self.assertAlmostEqual(p2.center, 35.222, tolerance=tolerance)
        self.assertAlmostEqual(p2.width, 0.02604, tolerance=tolerance)
    
    @unittest.expectedFailure
    def test_initial_pseudovoigt(self):
        # Does the class guess reasonable starting values for peak fitting
        # This specific peak originally guessed widths that are too large
        peakScan = XRDScan(corundum_path, phase=Corundum())
        df = peakScan.diffractogram[42.5:44]
        peak = XRDPeak(method="pseudo-voigt")
        guess = peak.guess_parameters(data=df.counts)
        # Should be two peaks present
        self.assertEqual(len(guess), 2)
        tolerance = 0.001
        # Returns two peaks for kα₁ and kα₂
        p1, p2 = guess
        self.assertAlmostEqual(p1.width_g, 0.02604, tolerance=tolerance)
        self.assertAlmostEqual(p1.width_c, 0.02604, tolerance=tolerance)
        self.assertAlmostEqual(p2.width_g, 0.02604, tolerance=tolerance)
        self.assertAlmostEqual(p2.width_c, 0.02604, tolerance=tolerance)
    
    @unittest.expectedFailure
    def test_peak_fit(self):
        """This particular peak was not fit properly. Let's see why."""
        peak = XRDPeak(reflection=Reflection('110', (2.59, 2.72)), method="gaussian")
        peakScan = XRDScan(os.path.join(TESTDIR, 'corundum.xye'),
                           phase=Corundum())
        df = peakScan.diffractogram
        bg = peakScan.refinement.refine_background(
            scattering_lengths=df.index,
            intensities=df['counts'].values
        )
        peak.fit(x=df.index, y=df['counts'].values - bg)
        # import matplotlib.pyplot as plt
        # plt.plot(df.index, df['intensities'].values - bg)
        # plt.show()
        fit_kalpha1 = peak.fit_list[0]
        fit_kalpha2 = peak.fit_list[1]
        print(fit_kalpha1.parameters)
        self.assertAlmostEqual(
            fit_kalpha1.parameters,
            fit_kalpha1.Parameters(height=30.133, center=37.774, width=0.023978)
        )
        self.assertAlmostEqual(
            fit_kalpha2.parameters,
            fit_kalpha2.Parameters(height=15.467, center=37.872, width=0.022393)
        )


class CubeTest(unittest.TestCase):
    def test_from_xy(self):
        """Can a set of x, y coords get the closest cube coords."""
        # Zero point
        cube = Cube.from_xy((0, 0), 1)
        self.assertEqual(cube, Cube(0, 0, 0))
        # Exact location
        cube = Cube.from_xy((0.5, math.sqrt(3)/2), unit_size=1)
        self.assertEqual(cube, Cube(1, 0, -1))
        # Rounding
        cube = Cube.from_xy((0.45, 0.9* math.sqrt(3)/2), unit_size=1)
        self.assertEqual(cube, Cube(1, 0, -1))


@unittest.expectedFailure
class LMOSolidSolutionTest(unittest.TestCase):
    def setUp(self):
        self.phase = CubicLMO()
        # self.map = XRDMap(scan_time=10,
        #                   qrange=(2.11, 3.76),
        #                   Phases=[CubicLMO],
        #                   background_phases=[Aluminum],
        #                   sample_name='test-sample')
        # self.map.reliability_normalizer = colors.Normalize(0.4, 0.8, clip=True)
    
    def test_metric(self):
        self.locus.refine_unit_cells()
        metric = self.locus.phases[0].unit_cell.a
        self.assertAlmostEqual(
            metric,
            8.192
        )
    
    def test_reliability_sample(self):
        self.locus.refine_background()
        self.locus.refine_scale_factors()
        reliability = self.locus.reliability
        self.assertTrue(
            reliability > 0.9,
            'Reliability {} is not > 0.9'.format(reliability)
        )
        signal_level = self.locus.signal_level
        self.assertAlmostEqual(
            signal_level,
            1.77,
            # 'Signal level {} is not < 0.1'.format(signal_level)
        )
    
    @unittest.expectedFailure
    def test_reliability_background(self):
        self.locus.load_diffractogram('test-sample-frames/LMO-background.plt')
        reliability = self.locus.reliability
        self.assertTrue(
            reliability < 0.1,
            'Reliability {} is not < 0.1'.format(reliability)
        )
    
    def test_reliability_noise(self):
        # Check that background noise gives low reliability
        self.locus.load_diffractogram('test-sample-frames/LMO-noise.plt')
        self.locus.refine_background()
        self.locus.refine_scale_factors()
        reliability = self.locus.reliability
        self.assertTrue(
            reliability < 0.1,
            'Reliability {} is not < 0.1'.format(reliability)
        )


class Refinement34IDETest(unittest.TestCase):
    """Tests that check for proper functioning of refinement with sample
    data from APS 34-ID-C beamline."""
    def setUp(self):
        self.map_ = XRDMap(hdf_filename=hdf_34IDE,
                           sample_name=group_34IDE,
                           Phases=[NCA])
    
    def test_unit_cell(self):
        idx = 85
        # self.map_.refine_mapping_data()
        with self.map_.store() as store:
            cellparams = store.cell_parameters[idx]
        expected_a = 2.88 # From GSAS-II refinement
        expected_c = 14.23 # From GSAS-II refinement


class XRDScanTest(unittest.TestCase):
    def setUp(self):
        self.xrd_scan = XRDScan(filename=corundum_path,
                                phase=Corundum)
    
    def test_remove_peak_from_df(self):
        xrd_scan = XRDScan(filename=corundum_path)
        peakRange = (2, 3)
        q = xrd_scan.scattering_lengths
        # peakIndex = df[peakRange[0]:peakRange[1]].index
        newq, intensities = remove_peak_from_df(x=q,
                                                y=xrd_scan.intensities,
                                                xrange=peakRange)
        self.assertEqual(
            len(newq),
            5404,
            'Expected new pattern to have length 5404 (got {})'.format(len(newq))
        )
        self.assertEqual(
            len(newq),
            len(intensities),
            'x and y are not the same length ({} vs {})'.format(len(newq), len(intensities)),
        )
    
    def test_contains_peak(self):
        """Method for determining if a given two_theta
        range is within the limits of the index."""
        x = self.xrd_scan.scattering_lengths
        # Completely inside range
        self.assertTrue(
            contains_peak(scattering_lengths=x, qrange=(1, 2))
        )
        # Completely outside range
        self.assertFalse(
            contains_peak(scattering_lengths=x, qrange=(0.2, 0.3))
        )
        # Partial overlap
        self.assertTrue(
            contains_peak(scattering_lengths=x, qrange=(5, 6))
        )


class UnitCellTest(unittest.TestCase):
    def test_init(self):
        unitCell = UnitCell(a=15, b=3, alpha=45)
        self.assertEqual(unitCell.a, 15)
        self.assertEqual(unitCell.b, 3)
        self.assertEqual(unitCell.alpha, 45)
    
    def test_setattr(self):
        """Does the unitcell give an error when passed crappy values."""
        # Negative unit cell parameter
        unitCell = UnitCell()
        with self.assertRaises(exceptions.UnitCellError):
            unitCell.a = -5
        with self.assertRaises(exceptions.UnitCellError):
            unitCell.alpha = -10


class CubicUnitCellTest(unittest.TestCase):
    def setUp(self):
        self.unit_cell = CubicUnitCell()
    
    def test_mutators(self):
        # Due to high symmetry, a=b=c
        self.unit_cell.a = 2
        self.assertEqual(self.unit_cell.b, 2)
        self.assertEqual(self.unit_cell.c, 2)
        with self.assertRaises(exceptions.UnitCellError):
            self.unit_cell.a = -5
        # and alpha=beta=gamma=90
        with self.assertRaises(exceptions.UnitCellError):
            self.unit_cell.alpha = 120
    
    def test_cell_parameters(self):
        self.assertEqual(
            self.unit_cell.cell_parameters,
            (1, )
        )
    
    def test_d_spacing(self):
        self.assertEqual(
            self.unit_cell.d_spacing((1, 1, 1)),
            math.sqrt(1/3)
        )


class HexagonalUnitCellTest(unittest.TestCase):
    def setUp(self):
        self.unit_cell = HexagonalUnitCell()
    
    def test_mutators(self):
        self.unit_cell.a = 3
        self.assertEqual(self.unit_cell.b, 3)
        self.assertNotEqual(self.unit_cell.c, 3)
        # Angles are fixed
        with self.assertRaises(exceptions.UnitCellError):
            self.unit_cell.alpha = 80
    
    def test_cell_parameters(self):
        self.unit_cell.a = 6.5
        self.unit_cell.c = 9
        self.assertEqual(
            self.unit_cell.cell_parameters,
            (6.5, 9)
        )
    
    def test_d_spacing(self):
        self.unit_cell.a = 1
        self.unit_cell.c = 2
        self.assertEqual(
            self.unit_cell.d_spacing((1, 2, 3)),
            math.sqrt(1/11.583333333333334)
        )


class ReflectionTest(unittest.TestCase):
    def test_hkl_to_tuple(self):
        newHkl = hkl_to_tuple((1, 1, 1))
        self.assertEqual(
            newHkl,
            (1, 1, 1)
        )
        newHkl = hkl_to_tuple('315')
        self.assertEqual(
            newHkl,
            (3, 1, 5)
        )
        newHkl = hkl_to_tuple('1 0 10')
        self.assertEqual(
            newHkl,
            (1, 0, 10)
        )
    
    def test_vague_hkl(self):
        # One of these indices is a 10 but it's not clear which one
        with self.assertRaises(exceptions.HKLFormatError):
            newHkl = hkl_to_tuple('1010')
        # This one's just nonsensical
        with self.assertRaises(exceptions.HKLFormatError):
            newHkl = hkl_to_tuple('1 d 10')
    
    def test_copy(self):
        ref0 = Reflection('111')
        ref1 = ref0.copy()
        # Change a value and ensure the copy stays the same
        ref0.intensity = 77
        self.assertNotEqual(ref1.intensity, ref0.intensity)


class PhaseTest(unittest.TestCase):
    def setUp(self):
        self.corundum_scan = XRDScan(filename='test-data-xrd/corundum.xye',
                                     phase=Corundum())
        self.phase = Corundum()
    
    def test_peak_by_hkl(self):
        reflection = self.phase.reflection_by_hkl('110')
        self.assertEqual(
            reflection.hkl,
            (1, 1, 0)
        )


class ExperimentalDataTest(unittest.TestCase):
    """
    These tests compare results to experimentally determined values.
    """
    def setUp(self):
        self.phase = Corundum()
    
    def test_predicted_peaks(self):
        # Predicted peaks up to (116) were calculated using celref
        # with the R-3C space group
        predicted_peaks = self.phase.predicted_peaks()
        celref_peaks = [ # (hkl, d, q)
            ('012', 3.4746228816945104, 1.8083071231360195),
            ('104', 2.5479680737754244, 2.4659591977812907),
            ('110', 2.3750000000000000, 2.6455517082861415),
            ('006', 2.1636666666666664, 2.9039525375964814),
            ('113', 2.0820345582756135, 3.0178102866762490),
            ('202', 1.9607287412929800, 3.2045153288446278),
            ('024', 1.7373114408472552, 3.6166142462720390),
            ('116', 1.5994489779586798, 3.9283436944631980),
            ('211', 1.5437700478607450, 4.0700266959359720),
            ('122', 1.5120305971116237, 4.1554617473893210),
            ('018', 1.5095406436761034, 4.1623160883422665),
            ('214', 1.4022018390633044, 4.4809421383849170),
            ('300', 1.3712068893253610, 4.5822299728022350),
            ('125', 1.3339201035011320, 4.7103160756691110),
            ('208', 1.2739840368877122, 4.9319183955625810),
            ('1010', 1.2380134239217553, 5.075215814119230),
        ]
        self.assertEqual(
            predicted_peaks[:len(celref_peaks)],
            celref_peaks
        )
    
    @unittest.skip('No refinement available')
    def test_mean_square_error(self):
        scan = XRDScan(filename=corundum_path,
                       phase=self.phase)
        scan.refinement.fit_peaks(scattering_lengths=scan.scattering_lengths,
                                  intensities=scan.intensities)
        rms_error = scan.refinement.peak_rms_error(phase=self.phase)
        # Check that the result is close to the value from celref
        diff = rms_error - 0.10492
        self.assertTrue(
            diff < 0.001
        )
    
    @unittest.expectedFailure
    def test_refine_corundum(self):
        # Results take from celref using corundum standard
        scan = XRDScan(filename=corundum_path,
                       phase=Corundum())
        residuals = scan.refinement.refine_unit_cells(
            quiet=True,
            scattering_lengths=scan.scattering_lengths,
            intensities=scan.intensities
        )
        # peaks = scan.phases[0].predicted_peaks()
        # plt.plot(scan.diffractogram.counts)
        # for peak in peaks:
        #     plt.axvline(peak.q, linestyle=":", alpha=0.5, color="C1")
        # plt.show()
        unit_cell_parameters = scan.phases[0].unit_cell.cell_parameters
        # Cell parameters taken from 1978a sample CoA
        self.assertAlmostEqual(
            unit_cell_parameters.a,
            4.758877,
        )
        self.assertAlmostEqual(
            unit_cell_parameters.c,
            12.992877
        )
        self.assertTrue(
            residuals < 0.03,
            'residuals ({}) too high'.format(residuals)
        )


# Unit tests for opening various XRD file formats
class XRDFileTestCase(unittest.TestCase):
    pass

class BrukerRawTestCase(XRDFileTestCase):
    """
    For data taken from Bruker instruments and save in various RAW
    file formats.
    """
    def setUp(self):
        self.adapter = BrukerRawFile(os.path.join(TESTDIR, 'test-data-xrd', 'corundum.raw'))
    
    def test_bad_file(self):
        badFile = os.path.join(TESTDIR, 'test-data-xrd', 'corundum.xye')
        with self.assertRaises(exceptions.FileFormatError):
            BrukerRawFile(badFile)
    
    @unittest.expectedFailure
    def test_sample_name(self):
        self.assertEqual(
            self.adapter.sample_name,
            'Format'
        )


class BrukerPltTestCase(unittest.TestCase):
    pltfile = os.path.join(TESTDIR, 'test-data-xrd', 'xrd-map-gadds', 'map-0.plt')
    def setUp(self):
        self.adapter = BrukerPltFile(self.pltfile)
    
    def test_intensities(self):
        Is = self.adapter.intensities()
        self.assertEqual(len(Is), 4001)
    
    def test_scattering_lengths(self):
        qs = self.adapter.scattering_lengths(wavelength=1.5418)
        self.assertEqual(len(qs), 4001)


class BrukerXyeTestCase(unittest.TestCase):
    def setUp(self):
        self.adapter = BrukerXyeFile(os.path.join(TESTDIR, 'test-data-xrd', 'corundum.xye'))
    
    def test_wavelength(self):
        self.assertAlmostEqual(
            self.adapter.wavelength,
            1.5418,
            places=3,
        )
    
    def test_scattering_lengths(self):
        """Check that two-theta values are converted to q."""
        q = self.adapter.scattering_lengths()
        # Check min and max values for scattering length
        self.assertAlmostEqual(min(q), 0.710, places=3)
        self.assertAlmostEqual(max(q), 5.239, places=3)


class BrukerBrmlTestCase(unittest.TestCase):
    def setUp(self):
        self.adapter = BrukerBrmlFile(os.path.join(TESTDIR, 'test-data-xrd', 'corundum.brml'))
    
    def test_wavelength(self):
        self.assertEqual(
            self.adapter.wavelength,
            1.5418,
        )
    
    def test_sample_name(self):
        self.assertEqual(
            self.adapter.sample_name,
            'Corundum (new Safety Board)'
        )
    
    def test_scattering_lengths(self):
        q = self.adapter.scattering_lengths()
        self.assertEqual(q[0], 0.71036599366565667)
    
    def test_counts(self):
        counts = self.adapter.intensities()
        self.assertEqual(counts[0], 122)


if __name__ == '__main__':
    unittest.main()
