import math, unittest
from io import StringIO
import os.path

import pandas as pd

from mapping import BaseSample, DummySample, Cube
from samples import LMOSolidSolution
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
        self.sample = LMOSolidSolution(center=(0.078,-30.688),
                                diameter=15,
                                collimator=0.5,
                                sample_name='test-sample')
        self.sample.create_scans()

    def test_metric(self):
        scan = self.sample.scans[0]
        df = scan.load_diffractogram()
        metric = scan.metric()
        self.assertEqual(
            metric,
            44.185
        )

    def test_reliability_sample(self):
        scan = self.sample.scans[0]
        scan.filename = 'LMO-sample'
        reliability = scan.reliability()
        self.assertTrue(
            reliability > 0.9,
            'Reliability {} is not > 0.9'.format(reliability)
       )

    def test_reliability_background(self):
        scan = self.sample.scans[0]
        scan.filename = 'LMO-background'
        reliability = scan.reliability()
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
        self.sample = BaseSample(center=(0, 0), diameter=12.7, rows=2,
                                 sample_name='slamfile-test')
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
        self.sample = BaseSample(center=(0, 0), diameter=12.7, collimator=0.5)
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
        sample = BaseSample(center=(0, 0), diameter=20, rows=5)
        self.assertEqual(sample.unit_size, 4/math.sqrt(3))

    def test_jinja_context(self):
        sample = BaseSample(center=(-10.5, 20.338), diameter=10, rows=4,
                            sample_name='LiMn2O4')
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
        self.scan = BaseSample.XRDScan(Cube(1, 0, -1), 'sample')
        self.scan.sample = DummySample()

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

if __name__ == '__main__':
    unittest.main()
