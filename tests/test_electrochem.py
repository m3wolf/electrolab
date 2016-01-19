# -*- coding: utf-8 -*-

import datetime as dt
import unittest
import os

from units import unit, predefined
predefined.define_units()

from electrochem.electrode import CathodeLaminate, CoinCellElectrode
from electrochem.galvanostatrun import GalvanostatRun
from electrochem import electrochem_units
from electrochem import biologic

testdir = os.path.join(os.path.dirname(__file__), 'testdata')
mptfile = os.path.join(testdir, 'NEI-C10_4cycles_C14.mpt')
# eclab-test-data.mpt
mprfile = os.path.join(testdir, 'NEI-C10_4cycles_C14.mpr')

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
        self.run = GalvanostatRun(mptfile, mass=0.022563)
        self.cycle = self.run.cycles[0]

    @unittest.expectedFailure
    def test_discharge_capacity(self):
        self.assertEqual(
            round(self.cycle.discharge_capacity(), 3),
            99.736
        )

class GalvanostatRunTest(ElectrolabTestCase):
    # Currently just tests import statement
    def test_import(self):
        GalvanostatRun(mptfile)

    def test_read_mass(self):
        run = GalvanostatRun(mptfile)
        self.assertEqual(run.mass, unit('g')(0.02265))

    def test_read_capacity(self):
        run = GalvanostatRun(mptfile)
        self.assertEqual(run.theoretical_capacity, unit('mAh')(3.357))

    def test_read_date(self):
        run = GalvanostatRun(mptfile)
        self.assertEqual(
            run.start_time,
            dt.datetime(2015, 7, 5, 15, 1, 23)
        )

    def test_read_current(self):
        run = GalvanostatRun(mptfile)
        # These are practically equal but assertEqual fails due to rounding in units package
        self.assertApproximatelyEqual(
            run.charge_current,
            unit('mA')(0.33570),
            tolerance=10**-15
        )
        self.assertApproximatelyEqual(
            run.discharge_current,
            unit('mA')(-335.70),
            tolerance=10**-15
        )

    def test_capacity_from_time(self):
        run = GalvanostatRun(mptfile)
        series = run.closest_datum(value=77090, label="time/s")
        self.assertApproximatelyEqual(series.capacity, 196.964615)


# class MPRTestCase(unittest.TestCase):

#     def setUp(self):
#         self.gvrun = GalvanostatRun(mprfile)

#     def test_output(self):
#         print(self.gvrun)
#         # self.assertEqual(
#         #     self.)

if __name__ == '__main__':
    unittest.main()
