import math, unittest

import pandas as pd

from samples import BaseSample, Scan, Cube
from cycler import GalvanostatRun

# class GalvanostatRunTest(unittest.TestCase):
#     # Currently just tests import statement
#     def test_import(self):
#         run = GalvanostatRun()

class CycleTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv('eclab-test-data.csv')
        self.run = GalvanostatRun(self.df, mass=0.022563)
        self.cycle = self.run.cycles[0]

    def test_discharge_capacity(self):
        self.assertEqual(
            self.cycle.discharge_capacity(),
            99.736007822541325
        )

class SlamFileTest(unittest.TestCase):

    def setUp(self):
        self.sample = BaseSample(center=(0, 0), diameter=12.7, rows=2)
        self.sample.two_theta_range = (50, 90)
        self.frame_step = 15
        self.frame_width = 20

    def test_number_of_frames(self):
        self.assertEqual(
            self.sample.get_number_of_frames(),
            3
        )
        # Check for values outside of limits
        self.sample.two_theta_range = (50, 200)
        self.assertRaises(
            ValueError,
            self.sample.get_number_of_frames
        )

    def test_theta2_start(self):
        self.assertEqual(
            self.sample.get_theta2_start(),
            7.5
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
        sample = BaseSample(center=(0, 0), diameter=10, rows=4,
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
            'LiMn2O4-3'
        )

class ScanTest(unittest.TestCase):
    def setUp(self):
        self.scan = Scan(Cube(1, 0, -1), 'sample')

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

    def test_unit_size(self):
        self.assertEqual(
            self.scan.xy_coords(2),
            (1, math.sqrt(3))
        )

if __name__ == '__main__':
    unittest.main()
