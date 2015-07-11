import math, unittest
from io import StringIO
import os.path

import pandas as pd

import electrolab as el
import xrdpeak
import materials
from materials import LMOSolidSolutionMaterial, CorundumMaterial
from mapping import Map, DummyMap, Cube, MapScan
# from xrd import Phase, hkl_to_tuple, Reflection, XRDScan, remove_peak_from_df
import xrd
import unitcell
import exceptions
from cycler import GalvanostatRun


class ElectrolabTestCase(unittest.TestCase):
    def assertApproximatelyEqual(self, actual, expected, tolerance=0.01, msg=None):
        """Assert that 'actual' is within relative 'tolerance' of 'expected'."""
        diff = abs(actual-expected)
        acceptable_diff = expected * tolerance
        if diff > acceptable_diff:
            msg = "{actual} is not close to {expected}"
            self.fail(msg=msg.format(actual=actual, expected=expected))


class PeakTest(unittest.TestCase):
    def test_split_parameters(self):
        peak = xrdpeak.Peak()
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
        peakScan = el.XRDScan('test-sample-frames/corundum.xye',
                              material=el.materials.CorundumMaterial())
        df = peakScan.diffractogram[34:36]
        peakFit = el.PeakFit()
        # Returns two peaks for kalpha1 and kalpha2
        p1, p2 = peakFit.initial_parameters(df.index, df.counts)
        self.assertEqual(
            p1,
            (426.60416666666703, 35.12325845859763, 0.02)
        )
        self.assertEqual(
            p2,
            (213.30208333333351, 35.22272924095449, 0.02)
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


class LMOSolidSolutionTest(unittest.TestCase):
    def setUp(self):
        self.material = LMOSolidSolutionMaterial()
        self.map = Map(scan_time=10,
                  two_theta_range=(10, 20))
        self.scan = MapScan(location=(0, 0), material=self.material,
                            xrd_map=self.map,
                            filebase="map-0")
        self.scan.load_diffractogram('test-sample-frames/LMO-sample-data.plt')

    def test_metric(self):
        df = self.scan.diffractogram
        metric = self.scan.metric
        self.assertEqual(
            metric,
            36.485
        )

    def test_reliability_sample(self):
        reliability = self.scan.reliability
        self.assertTrue(
            reliability > 0.9,
            'Reliability {} is not > 0.9'.format(reliability)
       )

    def test_reliability_background(self):
        self.scan.load_diffractogram('test-sample-frames/LMO-background.plt')
        reliability = self.scan.reliability
        self.assertTrue(
            reliability < 0.1,
            'Reliability {} is not < 0.1'.format(reliability)
        )

    def test_reliability_noise(self):
        # Check that background noise gives low reliability
        self.scan.load_diffractogram('test-sample-frames/LMO-noise.plt')
        reliability = self.scan.reliability
        self.assertTrue(
            reliability < 0.1,
            'Reliability {} is not < 0.1'.format(reliability)
        )

class CycleTest(unittest.TestCase):
    def setUp(self):
        # self.df = pd.read_csv()
        self.run = GalvanostatRun('eclab-test-data.mpt', mass=0.022563)
        self.cycle = self.run.cycles[0]

    def test_discharge_capacity(self):
        self.assertEqual(
            round(self.cycle.discharge_capacity(), 3),
            99.736
        )

class GalvanostatRunTest(unittest.TestCase):
    # Currently just tests import statement
    def test_import(self):
        run = GalvanostatRun('eclab-test-data.mpt')


class SlamFileTest(unittest.TestCase):

    def setUp(self):
        self.sample = Map(center=(0, 0), diameter=12.7,
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
        # Does passing a collimator diameter set the appropriate number of rows
        self.sample = Map(diameter=12.7, collimator=0.5)
        self.assertEqual(
            self.sample.rows,
            18
        )

    def test_theta2_start(self):
        self.assertEqual(
            self.sample.get_theta2_start(),
            11.25
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
            11.25
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
        self.sample.create_scans()
        self.assertEqual(
            self.sample.scans[8].cube_coords,
            Cube(2, 0, -2)
        )

    def test_coverage(self):
        halfMap = Map(collimator=2, coverage=0.25)
        self.assertEqual(halfMap.unit_size, 2 * math.sqrt(3))

    def test_cell_size(self):
        unitMap = Map(collimator=2)
        self.assertEqual(unitMap.unit_size, math.sqrt(3))

    def test_jinja_context(self):
        sample = Map(center=(-10.5, 20.338),
                     diameter=10,
                     sample_name='LiMn2O4',
                     scan_time=10,
                     two_theta_range=(10, 20))
        sample.create_scans()
        context = sample.get_context()
        self.assertEqual(
            len(context['scans']),
            len(sample.scans)
        )
        self.assertEqual(
            context['scans'][1]['x'],
            sample.scans[1].xy_coords(sample.unit_size)[0]
        )
        self.assertEqual(
            context['scans'][1]['y'],
            sample.scans[1].xy_coords(sample.unit_size)[1]
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

    def test_write_slamfile(self):
        directory = '{}-frames'.format(self.sample.sample_name)
        # Check that the directory does not already exist
        self.assertFalse(
            os.path.exists(directory),
            'Directory {} already exists, cannot test'.format(directory)
        )
        # Write the slamfile
        result = self.sample.write_slamfile(quiet=True)
        # Test if the correct things were created
        self.assertTrue(os.path.exists(directory))
        # Clean up
        os.remove('{directory}/{filename}.slm'.format(
            directory=directory,
            filename=self.sample.sample_name)
        )
        os.rmdir(directory)


class XRDScanTest(unittest.TestCase):
    def setUp(self):
        self.xrd_scan = xrd.XRDScan(filename='test-sample-frames/corundum.xye',
                                    material=CorundumMaterial())

    def test_remove_peak_from_df(self):
        xrd_scan = xrd.XRDScan(filename='test-sample-frames/map-0.plt')
        peakRange = (35, 40)
        df = xrd_scan.diffractogram
        peakIndex = df[peakRange[0]:peakRange[1]].index
        xrd.remove_peak_from_df(xrd.Reflection(peakRange, '000'), df)
        intersection = df.index.intersection(peakIndex)
        self.assertEqual(
            len(intersection),
            0,
            'Peak not removed ({0} remaining)'.format(len(intersection))
        )

    def test_filetypes(self):
        # Can the class determine the filetype and load it appropriately
        scan = xrd.XRDScan(filename='test-sample-frames/corundum.xye')

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



class TestUnitCell(unittest.TestCase):
    def test_init(self):
        unitCell = unitcell.UnitCell(a=15, b=3, alpha=45)
        self.assertEqual(unitCell.a, 15)
        self.assertEqual(unitCell.b, 3)
        self.assertEqual(unitCell.alpha, 45)

    def test_setattr(self):
        """Does the unitcell give an error when passed crappy values."""
        # Negative unit cell parameter
        unitCell = unitcell.UnitCell()
        with self.assertRaises(exceptions.UnitCellError):
            unitCell.a = -5
        with self.assertRaises(exceptions.UnitCellError):
            unitCell.alpha = -10


class TestCubicUnitCell(unittest.TestCase):
    def setUp(self):
        self.unit_cell = unitcell.CubicUnitCell()

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


class TestHexagonalUnitCell(unittest.TestCase):
    def setUp(self):
        self.unit_cell = unitcell.HexagonalUnitCell()

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


class MapScanTest(unittest.TestCase):
    def setUp(self):
        xrdMap = Map(scan_time=10,
                     two_theta_range=(10, 20))
        self.scan = MapScan(Cube(1, 0, -1), xrd_map=xrdMap, filebase="map-0",
                            material=materials.CorundumMaterial())
        self.scan.sample = DummyMap(scan_time=10,
                                    two_theta_range=(10, 20))

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
#             {'width': 569, 'height': 380},
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
            dataDict['filename'],
            scan.filename
        )
        self.assertEqual(
            dataDict['filebase'],
            scan.filebase
        )
        self.assertEqual(
            dataDict['metric'],
            scan.metric
        )
        self.assertEqual(
            dataDict['reliability'],
            scan.reliability
        )
        self.assertEqual(
            dataDict['spline'],
            scan.spline
        )


class MapTest(unittest.TestCase):
    def setUp(self):
        self.test_map = Map()

    def test_save(self):
        self.test_map.save()
        self.assertTrue(
            os.path.isfile('unknown.map')
        )
        os.remove('unknown.map')

class ReflectionTest(unittest.TestCase):
    def test_hkl_to_tuple(self):
        newHkl = xrd.hkl_to_tuple((1, 1, 1))
        self.assertEqual(
            newHkl,
            (1, 1, 1)
        )
        newHkl = xrd.hkl_to_tuple('315')
        self.assertEqual(
            newHkl,
            (3, 1, 5)
        )


class PhaseTest(unittest.TestCase):
    def setUp(self):
        self.phase = materials.CorundumPhase()
        self.corundum = materials.CorundumMaterial()
        self.corundum_scan = xrd.XRDScan(filename='test-sample-frames/corundum.xye',
                                         material=self.corundum)

    def test_peak_by_hkl(self):
        reflection = self.phase.reflection_by_hkl('110')
        self.assertEqual(
            reflection.hkl,
            (1, 1, 0)
        )

    def test_peak_list(self):
        phase = self.corundum.phase_list[0]
        phase.fit_peaks(scan=self.corundum_scan)
        peak_list = phase.peak_list
        two_theta_list = [peak.center_kalpha for peak in peak_list]
        hkl_list = [peak.reflection.hkl_string for peak in peak_list]
        self.assertEqual(
            two_theta_list,
            [25.599913304005099,
             35.178250906935716,
             37.790149818489454,
             41.709732482339412,
             43.388610036562113,
             52.594640340604649,
             57.54659705350258,
             61.353393926907451]
        )
        self.assertEqual(
            hkl_list,
            [reflection.hkl_string for reflection in self.phase.reflection_list]
        )


class ExperimentalDataTest(ElectrolabTestCase):
    """
    These tests compare results to experimentally determined values.
    """
    def setUp(self):
        self.material = materials.CorundumMaterial()
        self.phase = self.material.phase_list[0]

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
            ('117', 1.4617153753449086, 63.6591003395956)
        ]
        self.assertEqual(
            predicted_peaks,
            celref_peaks
        )

    def test_mean_square_error(self):
        scan = xrd.XRDScan(filename='test-sample-frames/corundum.xye',
                           material=self.material)
        scan.fit_peaks()
        rms_error = self.phase.peak_rms_error(scan=scan)
        # Check that the result is close to the value from celref
        diff = rms_error - 0.10492
        self.assertTrue(
            diff < 0.001
        )

    def test_refine_corundum(self):
        # Results take from celref using corundum standard
        scan = xrd.XRDScan(filename='test-sample-frames/corundum.xye',
                           material=self.material)
        residuals = self.phase.refine_unit_cell(scan=scan,
                                                quiet=True)
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
            residuals < 0.00288,
            'residuals ({}) too high'.format(residuals)
        )


if __name__ == '__main__':
    unittest.main()
