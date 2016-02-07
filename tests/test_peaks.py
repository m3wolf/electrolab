import unittest

import pandas as pd

from cases import ScimapTestCase
from peakfitting import Peak

# flake8: noqa

"""Module for testing generic peak fitting. Some techniques have more
specific tests (eg X-ray diffraction, X-ray microscopy."""

class GuessParameters(ScimapTestCase):

    def setUp(self):
        self.data = pd.Series([0, 3, 9, 3, 0])

    def test_single_peak(self):
        peak = Peak(num_peaks=1, method="Gaussian")
        guess = peak.guess_parameters(x=self.data.index,
                                      y=self.data.values)
        self.assertEqual(len(guess), 1)
        self.assertEqual(guess[0].height, 9)  # Height
        self.assertEqual(guess[0].center, 2)  # Center
        self.assertApproximatelyEqual(guess[0].width, 0.85)  # Width

    def test_two_peaks(self):
        pass


if __name__ == "__main__":
    unittest.main()
