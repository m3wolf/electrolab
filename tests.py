# -*- coding: utf-8 -*-

import math, unittest
import os.path

from matplotlib import colors
from units import unit, predefined
predefined.define_units()

import exceptions
import scimap
from phases.standards import Corundum, Aluminum
from phases.lmo import CubicLMO
from phases.unitcell import UnitCell, CubicUnitCell, HexagonalUnitCell
from mapping.map import Map, DummyMap, PeakPositionMap
from mapping.coordinates import Cube
from mapping.locus import Locus, cached_property
from xrd.map import XRDMap
from xrd.scan import XRDScan
from xrd.locus import XRDLocus
from xrd.peak import XRDPeak, PeakFit, remove_peak_from_df
from xrd.reflection import Reflection, hkl_to_tuple
from electrochem.electrode import CathodeLaminate, CoinCellElectrode
from electrochem.galvanostatrun import GalvanostatRun
from electrochem import electrochem_units
from adapters.bruker_raw_file import BrukerRawFile
from adapters.bruker_brml_file import BrukerBrmlFile
from refinement import fullprof, native

# Some phase definitions for testing
class LMOHighV(CubicLMO):
    unit_cell = CubicUnitCell(a=8.05)
    diagnostic_hkl = '333'
    reflection_list = [
        Reflection('333', (58.5, 59.3))
    ]


class LMOMidV(CubicLMO):
    unit_cell = CubicUnitCell(a=8.13)
    diagnostic_hkl = '333'
    reflection_list = [
        Reflection('333', (59.3, 59.9))
    ]


class ElectrolabTestCase(unittest.TestCase):
    def assertApproximatelyEqual(self, actual, expected, tolerance=0.01, msg=None):
        """Assert that 'actual' is within relative 'tolerance' of 'expected'."""
        # Check for lists
        if isinstance(actual, list) or isinstance(actual, tuple):
            for a, b in zip(actual, expected):
                self.assertApproximatelyEqual(a, b, tolerance=tolerance)
        else:
            diff = abs(actual-expected)
            acceptable_diff = abs(expected * tolerance)
            if diff > acceptable_diff:
                msg = "{actual} is not close to {expected}"
                self.fail(msg=msg.format(actual=actual, expected=expected))


class ElectrodeTest(ElectrolabTestCase):
    def setUp(self):
        self.laminate = CathodeLaminate(mass_active_material=0.9,
                                        mass_carbon=0.05,
                                        mass_binder=0.05,
                                        name="LMO-NEI")
        self.electrode = CoinCellElectrode(total_mass=unit('mg')(15),
                                           substrate_mass=unit('mg')(5),
                                           laminate=self.laminate,
                                           name="DummyElectrode",
                                           diameter=unit('mm')(12.7))

    def test_area(self):
        area_unit = unit('cm') * unit('cm')
        expected_area = area_unit(math.pi * (1.27/2)**2)
        self.assertEqual(self.electrode.area(), expected_area)

    def test_mass_loading(self):
        """Ensure the electrode can calculate the loading in mg/cm^2."""
        loading_units = unit('mg')/(unit('cm')*unit('cm'))
        area = math.pi * (1.27/2)**2
        expected = loading_units((15-5)*0.9 / area)
        self.assertEqual(self.electrode.mass_loading(), expected)


class PeakTest(ElectrolabTestCase):
    def test_split_parameters(self):
        peak = XRDPeak()
        # Put in some junk data so it will actually split
        peak.fit_list = ['a', 'b']
        fullParams = (1, 2, 3, 4, 5, 6)
        splitParams = peak.split_parameters(fullParams)
        self.assertEqual(
            splitParams,
            [(1, 2, 3), (4, 5, 6)]
        )

    def test_initial_parameters(self):
        # Does the class guess reasonable starting values for peak fitting
        peakScan = scimap.XRDScan('test-sample-frames/corundum.xye',
                              phase=Corundum())
        df = peakScan.diffractogram[34:36]
        peakFit = PeakFit()
        # Returns two peaks for kalpha1 and kalpha2
        p1, p2 = peakFit.initial_parameters(df.index, df.counts)
        tolerance = 0.001
        self.assertApproximatelyEqual(p1.height, 426.604, tolerance=tolerance)
        self.assertApproximatelyEqual(p1.center, 35.123, tolerance=tolerance)
        self.assertApproximatelyEqual(p1.width, 0.02, tolerance=tolerance)
        self.assertApproximatelyEqual(p2.height, 213.302, tolerance=tolerance)
        self.assertApproximatelyEqual(p2.center, 35.222, tolerance=tolerance)
        self.assertApproximatelyEqual(p2.width, 0.02, tolerance=tolerance)

    def test_peak_fit(self):
        """This particular peak was not fit properly. Let's see why."""
        peak = XRDPeak(reflection=Reflection('110', (37, 39)))
        peakScan = XRDScan('test-sample-frames/corundum.xye',
                           phase=Corundum())
        peakScan.refinement.refine_background()
        df = peakScan.refinement.subtracted[37:39]
        peak.fit(two_theta=df.index, intensity=df, method='gaussian')
        fit_kalpha1 = peak.fit_list[0]
        fit_kalpha2 = peak.fit_list[1]
        self.assertApproximatelyEqual(
            fit_kalpha1.parameters,
            fit_kalpha1.Parameters(height=30.133, center=37.774, width=0.023978)
        )
        self.assertApproximatelyEqual(
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


class LMOSolidSolutionTest(ElectrolabTestCase):
    def setUp(self):
        self.phase = CubicLMO()
        self.map = XRDMap(scan_time=10,
                                   two_theta_range=(30, 55),
                                   phases=[CubicLMO],
                                   background_phases=[Aluminum])
        self.map.reliability_normalizer = colors.Normalize(0.4, 0.8, clip=True)
        self.locus = self.map.loci[0]
        self.locus.load_diffractogram('test-sample-frames/LMO-sample-data.plt')

    def test_metric(self):
        self.locus.refine_unit_cells()
        metric = self.locus.phases[0].unit_cell.a
        self.assertApproximatelyEqual(
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
        self.assertApproximatelyEqual(
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


class XRDMapTest(ElectrolabTestCase):
    def setUp(self):
        self.map = XRDMap()


class NativeRefinementTest(ElectrolabTestCase):
    def setUp(self):
        self.scan = XRDScan(
            'test-sample-frames/lmo-two-phase.brml',
            phases=[LMOHighV(), LMOMidV()], refinement=native.NativeRefinement
        )
        self.refinement = self.scan.refinement

    def test_two_phase_ratio(self):
        refinement = self.refinement
        refinement.refine_scale_factors()
        self.assertTrue(
            refinement.is_refined['scale_factors']
        )
        self.assertApproximatelyEqual(
            self.scan.phases[0].scale_factor,
            275
        )
        self.assertApproximatelyEqual(
            self.scan.phases[1].scale_factor,
            205
        )

    def test_peak_area(self):
        reflection = LMOMidV().diagnostic_reflection
        self.assertApproximatelyEqual(
            self.refinement.net_area(reflection.two_theta_range),
            205
        )

    @unittest.expectedFailure
    def test_peak_fwhm(self):
        """Method for computing full-width at half max of a peak."""
        result = self.xrd_scan.refinement.fwhm(twotheta=52.5)
        self.assertApproximatelyEqual(
            result,
            0.0594
        )

    def test_peak_list(self):
        corundum_scan = XRDScan(filename='test-sample-frames/corundum.xye',
                                phase=Corundum())
        # corundum_scan.refinement.fit_peaks(scan=self.corundum_scan)
        peak_list = corundum_scan.refinement.peak_list
        two_theta_list = [peak.center_kalpha for peak in peak_list]
        hkl_list = [peak.reflection.hkl_string for peak in peak_list]
        self.assertApproximatelyEqual(
            two_theta_list,
            [25.599913304005099,
             35.178250906935716,
             37.790149818489454,
             41.709732482339412,
             43.388610036562113,
             52.594640340604649,
             57.54659705350258],
            tolerance=0.001
        )
        self.assertEqual(
            hkl_list,
            [reflection.hkl_string for reflection in Corundum.reflection_list]
        )


class ElectrochemUnitsTest(ElectrolabTestCase):
    """Check that various values for current, capacity, etc are compatible."""
    def setUp(self):
        self.mAh = electrochem_units.mAh
        self.hour = electrochem_units.hour
        self.mA = unit('mA')
        self.uA = unit('µA')

    def test_milli_micro_amps(self):
        time = self.mAh(10) / self.uA(1000)
        self.assertApproximatelyEqual(self.hour(time), self.hour(10),
                                      tolerance=10**-10)


class CycleTest(unittest.TestCase):
    def setUp(self):
        self.run = GalvanostatRun('electrochem/eclab-test-data.mpt',
                                  mass=0.022563)
        self.cycle = self.run.cycles[0]

    def test_discharge_capacity(self):
        self.assertEqual(
            round(self.cycle.discharge_capacity(), 3),
            99.736
        )

class GalvanostatRunTest(ElectrolabTestCase):
    # Currently just tests import statement
    def test_import(self):
        GalvanostatRun('electrochem/eclab-test-data.mpt')

    def test_read_mass(self):
        run = GalvanostatRun('electrochem/eclab-test-data.mpt')
        self.assertEqual(run.mass, unit('g')(0.02253))

    def test_read_capacity(self):
        run = GalvanostatRun('electrochem/eclab-test-data.mpt')
        self.assertEqual(run.theoretical_capacity, unit('mAh')(3.340))

    def test_read_current(self):
        run = GalvanostatRun('electrochem/eclab-test-data.mpt')
        # These are practically equal but assertEqual fails due to rounding in units package
        self.assertApproximatelyEqual(
            run.charge_current,
            unit('mA')(0.334),
            tolerance=10**-15
        )
        self.assertApproximatelyEqual(
            run.discharge_current,
            unit('mA')(-0.334),
            tolerance=10**-15
        )

class SlamFileTest(unittest.TestCase):

    def setUp(self):
        self.sample = XRDMap(center=(0, 0), diameter=12.7,
                          sample_name='slamfile-test',
                          scan_time=5, two_theta_range=(10, 20))
        self.sample.two_theta_range = (50, 90)

    def test_number_of_frames(self):
        self.assertEqual(
            self.sample.get_number_of_frames(),
            2
        )
        # Check for values outside of limits
        self.sample.two_theta_range = (50, 200)
        self.assertRaises(
            ValueError,
            self.sample.get_number_of_frames
        )

    def test_rows(self):
        # Does passing a resolution set the appropriate number of rows
        self.sample = XRDMap(diameter=12.7, resolution=0.5)
        self.assertEqual(
            self.sample.rows,
            18
        )

    def test_theta2_start(self):
        self.assertEqual(
            self.sample.get_theta2_start(),
            10
        )

    def test_theta1(self):
        self.assertEqual(
            self.sample.get_theta1(),
            50
        )
        # Check for values outside the limits
        self.sample.two_theta_range = (-5, 50)
        self.assertRaises(
            ValueError,
            self.sample.get_theta1
        )
        # Check for values outside theta1 max
        self.sample.two_theta_range = (60, 90)
        self.assertEqual(
            self.sample.get_theta1(),
            50
        )

    def test_small_angles(self):
        """
        See what happens when the 2theta angle is close to the max X-ray
        source angle.
        """
        self.sample.two_theta_range = (47.5, 62.5)
        self.assertEqual(
            self.sample.get_theta1(),
            47.5
        )
        self.assertEqual(
            self.sample.get_theta2_start(),
            10
        )

    def test_path(self):
        results_list = []
        for coords in self.sample.path(2):
            results_list.append(coords)
        self.assertEqual(
            results_list,
            [Cube(0, 0, 0),
             Cube(1, 0, -1),
             Cube(0, 1, -1),
             Cube(-1, 1, 0),
             Cube(-1, 0, 1),
             Cube(0, -1, 1),
             Cube(1, -1, 0)]
        )
        self.sample.create_loci()
        self.assertEqual(
            self.sample.loci[8].cube_coords,
            Cube(2, 0, -2)
        )

    def test_coverage(self):
        halfMap = XRDMap(collimator=2, coverage=0.25)
        self.assertEqual(halfMap.unit_size, 2 * math.sqrt(3))

    def test_cell_size(self):
        unitMap = XRDMap(collimator=2)
        self.assertEqual(unitMap.unit_size, math.sqrt(3))

    def test_jinja_context(self):
        sample = XRDMap(center=(-10.5, 20.338),
                     diameter=10,
                     sample_name='LiMn2O4',
                     scan_time=10,
                     two_theta_range=(10, 20))
        sample.create_loci()
        context = sample.context()
        self.assertEqual(
            len(context['scans']),
            len(sample.loci)
        )
        self.assertEqual(
            context['scans'][1]['x'],
            sample.loci[1].xy_coords(sample.unit_size)[0]
        )
        self.assertEqual(
            context['scans'][1]['y'],
            sample.loci[1].xy_coords(sample.unit_size)[1]
        )
        self.assertEqual(
            context['scans'][3]['filename'],
            'map-3'
        )
        self.assertEqual(
            context['xoffset'],
            -10.5
        )
        self.assertEqual(
            context['yoffset'],
            20.338
        )
        # Flood and spatial files to load
        self.assertEqual(
            context['flood_file'],
            '1024_020._FL'
        )
        self.assertEqual(
            context['spatial_file'],
            '1024_020._ix'
        )

    def test_write_slamfile(self):
        directory = '{}-frames'.format(self.sample.sample_name)
        # Check that the directory does not already exist
        self.assertFalse(
            os.path.exists(directory),
            'Directory {} already exists, cannot test'.format(directory)
        )
        # Write the slamfile
        self.sample.write_script(quiet=True)
        # Test if the correct things were created
        self.assertTrue(os.path.exists(directory))
        # Clean up
        os.remove('{directory}/{filename}.slm'.format(
            directory=directory,
            filename=self.sample.sample_name)
        )
        os.rmdir(directory)


class XRDScanTest(ElectrolabTestCase):
    def setUp(self):
        self.xrd_scan = XRDScan(filename='test-sample-frames/corundum.xye',
                                phase=Corundum)

    def test_remove_peak_from_df(self):
        xrd_scan = XRDScan(filename='test-sample-frames/map-0.plt')
        peakRange = (35, 40)
        df = xrd_scan.diffractogram
        peakIndex = df[peakRange[0]:peakRange[1]].index
        remove_peak_from_df(Reflection('000', peakRange), df)
        intersection = df.index.intersection(peakIndex)
        self.assertEqual(
            len(intersection),
            0,
            'Peak not removed ({0} remaining)'.format(len(intersection))
        )

    def test_filetypes(self):
        # Can the class determine the filetype and load it appropriately
        XRDScan(filename='test-sample-frames/corundum.xye')

    def test_contains_peak(self):
        """Method for determining if a given two_theta
        range is within the limits of the index."""
        # Completely inside range
        self.assertTrue(
            self.xrd_scan.contains_peak(two_theta_range=(20, 23))
        )
        # Completely outside range
        self.assertFalse(
            self.xrd_scan.contains_peak(two_theta_range=(2, 3))
        )
        # Partial overlap
        self.assertTrue(
            self.xrd_scan.contains_peak(two_theta_range=(79, 81))
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


class XRDLocusTest(unittest.TestCase):
    def setUp(self):
        xrd_map = XRDMap(scan_time=10,
                         two_theta_range=(10, 20))
        self.scan = XRDLocus(location=Cube(1, 0, -1),
                             parent_map=xrd_map,
                             filebase="map-0")

    def test_xy_coords(self):
        self.scan.cube_coords = Cube(1, -1, 0)
        self.assertEqual(
            self.scan.xy_coords(1),
            (1, 0)
        )
        self.scan.cube_coords = Cube(1, 0, -1)
        self.assertEqual(
            self.scan.xy_coords(1),
            (0.5, math.sqrt(3)/2)
        )
        self.scan.cube_coords = Cube(0, 1, -1)
        self.assertEqual(
            self.scan.xy_coords(1),
            (-0.5, math.sqrt(3)/2)
        )
        self.scan.cube_coords = Cube(1, -2, 1)
        self.assertEqual(
            self.scan.xy_coords(1),
            (1.5, -math.sqrt(3)/2)
        )
        self.scan.cube_coords = Cube(2, 0, -2)
        self.assertEqual(
            self.scan.xy_coords(1),
            (1, math.sqrt(3))
        )

    def test_pixel_coords(self):
        self.assertEqual(
            self.scan.pixel_coords(height=1000, width=1000),
            {'width': 553, 'height': 408},
        )

    def test_unit_size(self):
        self.assertEqual(
            self.scan.xy_coords(2),
            (1, math.sqrt(3))
        )

    def test_data_dict(self):
        scan = self.scan
        dataDict = scan.data_dict
        self.assertEqual(
            dataDict['diffractogram'],
            scan.diffractogram
        )
        self.assertEqual(
            dataDict['cube_coords'],
            tuple(scan.cube_coords)
        )
        self.assertEqual(
            dataDict['filebase'],
            scan.filebase
        )
        self.assertEqual(
            dataDict['metric'],
            scan.metric
        )


class XRDMapTest(unittest.TestCase):
    def setUp(self):
        self.test_map = XRDMap(phases=[Corundum], sample_name='test-sample')
        self.savefile = 'test-sample.map'

    def tearDown(self):
        try:
            os.remove(self.savefile)
        except FileNotFoundError:
            pass

    def test_set_phases(self):
        """Verify that phases are passed to all the necessary composited objects."""
        new_map = XRDMap(phases=[Corundum])
        self.assertTrue(
            isinstance(new_map.loci[0].phases[0], Corundum)
        )

    def test_pass_filename(self):
        self.assertEqual(
            self.test_map.loci[0].filename,
            self.test_map.loci[0].xrdscan.filename
        )

    def test_save_map(self):
        # Set some new data
        self.test_map.diameter = 5
        self.test_map.coverage = 0.33
        self.test_map.save()
        # Make sure savefile was created
        self.assertTrue(
            os.path.isfile(self.savefile)
        )
        # Load from file
        new_map = XRDMap()
        new_map.load(filename=self.savefile)
        self.assertEqual(new_map.diameter, self.test_map.diameter)
        self.assertEqual(new_map.coverage, self.test_map.coverage)

    def test_save_loci(self):
        """Does the save routine properly save the loci list."""
        original_locus = self.test_map.loci[0]
        original_locus.metric = 200
        original_locus.filebase = 'nonsense'
        original_locus.cube_coords = Cube(20, 20, 20)
        original_locus.diffractogram = 'Gibberish'
        self.test_map.save()
        new_map = XRDMap()
        new_map.load(self.savefile)
        new_locus = new_map.loci[0]
        self.assertEqual(new_locus.metric, original_locus.metric)
        self.assertEqual(new_locus.filebase, original_locus.filebase)
        self.assertEqual(new_locus.cube_coords, original_locus.cube_coords)
        self.assertEqual(new_locus.diffractogram, original_locus.diffractogram)

    def test_save_refinement(self):
        original = self.test_map.loci[0].refinement
        original.spline = (1, 3, 5)
        self.test_map.save()
        new_map = XRDMap()
        new_map.load(self.savefile)
        new_refinement = new_map.loci[0].refinement
        self.assertEqual(new_refinement.spline, original.spline)

    def test_save_phases(self):
        original = self.test_map.loci[0].phases[0]
        original.scale_factor = 100
        self.test_map.save()
        new_map = XRDMap(phases=[Corundum])
        new_map.load(self.savefile)
        new_phase = new_map.loci[0].phases[0]
        self.assertEqual(new_phase.scale_factor, original.scale_factor)


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


class PhaseTest(ElectrolabTestCase):
    def setUp(self):
        self.corundum_scan = XRDScan(filename='test-sample-frames/corundum.xye',
                                     phase=Corundum())
        self.phase = Corundum()

    def test_peak_by_hkl(self):
        reflection = self.phase.reflection_by_hkl('110')
        self.assertEqual(
            reflection.hkl,
            (1, 1, 0)
        )

    def test_unitcell_copy(self):
        """
        Stems from a bug where setting a parameter on one unit cell changes
        another.
        """
        xrdmap = XRDMap(scan_time=10,
                        two_theta_range=(30, 55),
                        phases=[CubicLMO],
                        background_phases=[Aluminum])
        phase1 = xrdmap.loci[0].phases[0]
        phase2 = xrdmap.loci[1].phases[0]
        # Changing parameter on one phase should not change the other
        phase1.unit_cell.a = 8
        phase2.unit_cell.a = 5
        self.assertIsNot(phase1.unit_cell, phase2.unit_cell)
        self.assertNotEqual(phase1.unit_cell.a, 5, 'Unit cells are coupled')


class CachingTest(ElectrolabTestCase):
    def test_cached_property(self):
        # Simple class to test caching
        class Adder():
            a = 1
            b = 2
            @cached_property
            def added(self):
                return self.a + self.b
        adder = Adder()
        self.assertEqual(adder.added, 3)
        # Now change an attribute and see if the cached value is returned
        adder.a = 3
        self.assertEqual(adder.added, 3)
        # Delete the cached value and see if a new value is computer
        del adder.added
        self.assertEqual(adder.added, 5)


class ExperimentalDataTest(ElectrolabTestCase):
    """
    These tests compare results to experimentally determined values.
    """
    def setUp(self):
        self.phase = Corundum()

    def test_predicted_peak_positions(self):
        # Predicted peaks were calculated using celref with the R-3C space group
        predicted_peaks = self.phase.predicted_peak_positions(wavelength=1.5418)
        celref_peaks = [
            ('012', 3.4746228816945104, 25.637288649553085),
            ('104', 2.5479680737754244, 35.22223164557721),
            ('110', 2.375, 37.88141047624646),
            ('006', 2.1636666666666664, 41.74546075011751),
            ('113', 2.0820345582756135, 43.46365474219995),
            ('024', 1.7373114408472552, 52.68443192186963),
            ('116', 1.5994489779586798, 57.62940019834231),
        ]
        self.assertEqual(
            predicted_peaks,
            celref_peaks
        )

    def test_mean_square_error(self):
        scan = XRDScan(filename='test-sample-frames/corundum.xye',
                       phase=self.phase)
        scan.refinement.fit_peaks()
        rms_error = scan.refinement.peak_rms_error(phase=self.phase)
        # Check that the result is close to the value from celref
        diff = rms_error - 0.10492
        self.assertTrue(
            diff < 0.001
        )

    def test_refine_corundum(self):
        # Results take from celref using corundum standard
        scan = XRDScan(filename='test-sample-frames/corundum.xye',
                       phase=Corundum())
        residuals = scan.refinement.refine_unit_cells(quiet=True)
        unit_cell_parameters = self.phase.unit_cell.cell_parameters
        # Cell parameters taken from 1978a sample CoA
        self.assertApproximatelyEqual(
            unit_cell_parameters.a,
            4.758877,
        )
        self.assertApproximatelyEqual(
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
        self.adapter = BrukerRawFile('test-sample-frames/corundum.raw')

    def test_bad_file(self):
        badFile = 'test-sample-frames/corundum.xye'
        with self.assertRaises(exceptions.FileFormatError):
            BrukerRawFile(badFile)
    @unittest.expectedFailure
    def test_sample_name(self):
        self.assertEqual(
            self.adapter.sample_name,
            'Format'
        )


class BrukerBrmlTestCase(unittest.TestCase):
    def setUp(self):
        self.adapter = BrukerBrmlFile('test-sample-frames/corundum.brml')

    def test_sample_name(self):
        self.assertEqual(
            self.adapter.sample_name,
            'Corundum (new Safety Board)'
        )

    def test_diffractogram(self):
        importedDf = self.adapter.dataframe
        self.assertTrue(
            'counts' in importedDf.columns
        )


class FullProfProfileTest(ElectrolabTestCase):
    def setUp(self):
        # Base parameters determine from manual refinement
        class FPCorundum(fullprof.FullProfPhase, Corundum):
            unit_cell = HexagonalUnitCell(a=4.758637, c=12.991814)
            u = 0.00767
            v = -0.003524
            w = 0.002903
            x = 0.001124
            eta = 0.511090
            isotropic_temp = 33.314
        self.scan = scimap.XRDScan('test-sample-frames/corundum.brml',
                               phase=FPCorundum())
        self.refinement = fullprof.ProfileMatch(scan=self.scan)
        self.refinement.zero = -0.003820
        self.refinement.displacement = 0.0012
        self.refinement.bg_coeffs = [129.92, -105.82, 108.32, 151.85, -277.55, 91.911]
        # self.refinement.keep_temp_files = True

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
        self.assertApproximatelyEqual(
            self.refinement.bg_coeffs,
            [132.87, -35.040, -5.6920, 0, 0, 0]
        )

    def test_refine_displacement(self):
        # Set sample displacement to something wrong
        self.refinement.displacement = 0
        self.refinement.refine_displacement()
        self.assertTrue(
            0 < self.refinement.chi_squared < 10,
            'Χ² is too high: {}'.format(self.refinement.chi_squared)
        )
        # Based on manual refinement in fullprof
        self.assertApproximatelyEqual(
            self.refinement.displacement,
            0.0054
        )

    def test_refine_unit_cell(self):
        # Set unit cell parameters off by a little bit
        phase = self.scan.phases[0]
        phase.unit_cell.a = 4.75
        phase.unit_cell.c = 12.982
        self.refinement.refine_unit_cells()
        self.assertTrue(self.refinement.is_refined['unit_cells'])
        self.assertTrue(self.refinement.chi_squared < 10)
        self.assertApproximatelyEqual(phase.unit_cell.a, 4.758637,
                                      tolerance=0.001)
        self.assertApproximatelyEqual(phase.unit_cell.c, 12.991814,
                                      tolerance=0.001)


class FullProfLmoTest(ElectrolabTestCase):
    """Check refinement using data from LiMn2O4 ("NEI")"""
    def setUp(self):
        # Base parameters determine from manual refinement
        class LMOHighV(fullprof.FullProfPhase, CubicLMO):
            unit_cell = CubicUnitCell(a=8.052577)
            isotropic_temp = 0.19019
            u = -0.000166
            v = 0.120548
            w = 0.003580
            I_g = 0.000142
            eta = 0.206420
            x = 0.007408
        class LMOMidV(fullprof.FullProfPhase, CubicLMO):
            unit_cell = CubicUnitCell(a=8.122771)
            isotropic_temp = -0.45434
            u = 0.631556
            v = -0.115778
            w = 0.019247
            I_g = -0.000539
            eta = 0.923930
            x = -0.006729
        self.scan = scimap.XRDScan('test-sample-frames/lmo-two-phase.brml',
                               phases=[LMOHighV(), LMOMidV()])
        self.refinement = fullprof.ProfileMatch(scan=self.scan)
        # Base parameters determined by manual refinement
        self.refinement.bg_coeffs = [71.297, -50.002, 148.13, -150.13, -249.84, 297.01]
        self.refinement.zero = 0.044580
        self.refinement.displacement = 0.000320
        self.refinement.transparency = -0.00810

    def test_scale_factors(self):
        self.refinement.refine_scale_factors()
        self.assertTrue(
            self.refinement.is_refined['scale_factors']
        )
        self.assertTrue(
            self.refinement.chi_squared < 10
        )
        self.assertApproximatelyEqual(
            self.scan.phases[0].scale_factor,
            37.621
        )
        self.assertApproximatelyEqual(
            self.scan.phases[1].scale_factor,
            40.592
        )


if __name__ == '__main__':
    unittest.main()
