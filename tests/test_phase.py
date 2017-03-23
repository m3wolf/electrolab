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
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import matplotlib.pyplot as plt

from scimap import XRDScan, standards


class NativeRefinementTest(unittest.TestCase):

    @unittest.expectedFailure
    def test_remove_peaks(self):
        corundum = standards.Corundum()
        # Get sample data from Mew XRD
        scan = XRDScan(filename="test-data-xrd/corundum.brml",
                       phase=corundum)
        df = scan.diffractogram
        q = df.index
        old_I = df.counts
        # Remove expected XRD peaks
        new_I = corundum.remove_peaks(q, old_I)
        # Check that the result is the same shape as the input data
        self.assertEqual(old_I.shape, new_I.shape)
        # Spot-check a few intensity values against previously verified results
        assert False, "TODO: Write a test for spot-checking the removed peaks"
