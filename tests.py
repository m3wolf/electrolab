import math, unittest
from io import StringIO
import os.path

import pandas as pd

import materials
from materials import LMOSolidSolutionMaterial, CorundumMaterial
from mapping import Map, DummyMap, Cube, MapScan
from xrd import Phase, hkl_to_tuple, Reflection, XRDScan, remove_peak_from_df
import xrd
from cycler import GalvanostatRun


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
        df = self.scan.diffractogram()
        metric = self.scan.metric()
        self.assertEqual(
            metric,
            36.485
        )

    def test_reliability_sample(self):
        reliability = self.scan.reliability()
        self.assertTrue(
            reliability > 0.9,
            'Reliability {} is not > 0.9'.format(reliability)
       )

    def test_reliability_background(self):
        self.scan.load_diffractogram('test-sample-frames/LMO-background.plt')
        reliability = self.scan.reliability()
        self.assertTrue(
            reliability < 0.1,
            'Reliability {} is not < 0.1'.format(reliability)
        )

    def test_reliability_noise(self):
        # Check that background noise gives low reliability
        self.scan.load_diffractogram('test-sample-frames/LMO-noise.plt')
        reliability = self.scan.reliability()
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
        self.sample = Map(center=(0, 0), diameter=12.7, rows=2,
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

    def test_collimator(self):
        # Does passing a collimator diameter set the appropriate number of rows
        self.sample = Map(center=(0, 0),
                          diameter=12.7,
                          collimator=0.5,
                          scan_time=10,
                          two_theta_range=(10, 20))
        self.assertEqual(
            self.sample.rows,
            13
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
        for coords in self.sample.path(1):
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

    def test_cell_size(self):
        sample = Map(center=(0, 0),
                     diameter=20,
                     rows=5,
                     scan_time=10,
                     two_theta_range=(10, 20))
        self.assertEqual(sample.unit_size, 4/math.sqrt(3))

    def test_jinja_context(self):
        sample = Map(center=(-10.5, 20.338),
                     diameter=10,
                     rows=4,
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
        result = self.sample.write_slamfile()
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
        self.xrd_scan = XRDScan(filename='test-sample-frames/corundum.xye',
                                material=CorundumMaterial())

    def test_remove_peak_from_df(self):
        xrd_scan = XRDScan(filename='test-sample-frames/map-0.plt')
        peakRange = (35, 40)
        df = xrd_scan.diffractogram()
        peakIndex = df[peakRange[0]:peakRange[1]].index
        remove_peak_from_df(Reflection(peakRange, '000'), df)
        intersection = df.index.intersection(peakIndex)
        self.assertEqual(
            len(intersection),
            0,
            'Peak not removed ({0} remaining)'.format(len(intersection))
        )

    def test_filetypes(self):
        # Can the class determine the filetype and load it appropriately
        scan = XRDScan(filename='test-sample-frames/corundum.xye')

    def test_contains_peak(self):
        """Method for determining if a given two_theta
        range is within the limits of the index."""
        # Completely inside range
        self.assertTrue(
            self.xrd_scan.contains_peak(peak=(20, 23))
        )
        # Completely outside range
        self.assertFalse(
            self.xrd_scan.contains_peak(peak=(2, 3))
        )
        # Partial overlap
        self.assertTrue(
            self.xrd_scan.contains_peak(peak=(79, 81))
        )

    def test_peak_list(self):
        peak_list = self.xrd_scan.peak_list
        self.assertEqual(
            peak_list,
            [25.575146738439802,
             35.151143238879698,
             37.777643879875498,
             41.671405530534699,
             43.347460414282999,
             52.545322581194299,
             57.491728457622202]
        )

    def test_refine_unit_cell(self):
        deviation = self.xrd_scan.refine_cell_parameters()
        self.assertEqual(
            self.xrd_scan.cell_parameters,
            (4.7628, 4.7628, 13.0045, 90, 90, 120)
        )
        self.assertEqual(deviation, 0.00288)


class TestUnitCell(unittest.TestCase):
    def test_init(self):
        unitCell = xrd.UnitCell(a=15, b=3, alpha=45)
        self.assertEqual(unitCell.a, 15)
        self.assertEqual(unitCell.b, 3)
        self.assertEqual(unitCell.alpha, 45)

    def test_setattr(self):
        """Does the unitcell give an error when passed crappy values."""
        # Negative unit cell parameter
        unitCell = xrd.UnitCell()
        with self.assertRaises(xrd.UnitCellError):
            unitCell.a = -5
        with self.assertRaises(xrd.UnitCellError):
            unitCell.alpha = -10

class TestCubicUnitCell(unittest.TestCase):
    def setUp(self):
        self.unit_cell = xrd.CubicUnitCell()

    def test_mutators(self):
        # Due to high symmetry, a=b=c
        self.unit_cell.a = 2
        self.assertEqual(self.unit_cell.b, 2)
        self.assertEqual(self.unit_cell.c, 2)
        with self.assertRaises(xrd.UnitCellError):
            self.unit_cell.a = -5
        # and alpha=beta=gamma=90
        with self.assertRaises(xrd.UnitCellError):
            self.unit_cell.alpha = 120

    def test_cell_parameters(self):
        self.assertEqual(
            self.unit_cell.cell_parameters,
            (1, )
        )


class TestHexagonalUnitCell(unittest.TestCase):
    def setUp(self):
        self.unit_cell = xrd.HexagonalUnitCell()

    def test_mutators(self):
        self.unit_cell.a = 3
        self.assertEqual(self.unit_cell.b, 3)
        self.assertNotEqual(self.unit_cell.c, 3)
        # Angles are fixed
        with self.assertRaises(xrd.UnitCellError):
            self.unit_cell.alpha = 80

    def test_cell_parameters(self):
        self.unit_cell.a = 6.5
        self.unit_cell.c = 9
        self.assertEqual(
            self.unit_cell.cell_parameters,
            (6.5, 9)
        )


class MapScanTest(unittest.TestCase):
    def setUp(self):
        xrdMap = Map(scan_time=10,
                     two_theta_range=(10, 20))
        self.scan = MapScan(Cube(1, 0, -1), xrd_map=xrdMap, filebase="map-0")
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
            {'width': 569, 'height': 380},
        )

    def test_unit_size(self):
        self.assertEqual(
            self.scan.xy_coords(2),
            (1, math.sqrt(3))
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


class PhaseTest(unittest.TestCase):
    def setUp(self):
        self.phase = Phase(
            reflection_list = [Reflection(two_theta_range=(0, 1), hkl='111')]
        )

    def test_peak_by_hkl(self):
        reflection = self.phase.reflection_by_hkl('111')
        self.assertEqual(
            reflection.hkl,
            (1, 1, 1)
        )


class ExperimentalDataTest(unittest.TestCase):
    """
    These tests compare results to experimentally determined values.
    """
    def test_predicted_peak_positions(self):
        # Predicted peaks were calculated using celref with the R-3C space group
        material = materials.CorundumMaterial()
        phase = material.phase_list[0]
        predicted_peaks = phase.predicted_peak_positions()
        celref_peaks = pd.DataFrame(
            [
                ('012', 3.4746, 25.637),
                ('104', 2.5480, 35.222),
                ('110', 2.3750, 37.881),
                ('006', 2.1637, 41.745),
                ('113', 2.0820, 43.464),
                ('024', 1.7373, 52.684),
                ('116', 1.5994, 57.629)
            ], columns=('hkl', 'd', '2theta')
        )
        self.assertEqual(
            predicted_peaks,
            celref_peaks
        )

if __name__ == '__main__':
    unittest.main()
