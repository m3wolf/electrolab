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
# from scimap import fit_background


class NativeRefinementTest(unittest.TestCase):

    @unittest.expectedFailure
    def test_fit_background(self):
        corundum = standards.Corundum()
        # Get sample data from Mew XRD
        scan = XRDScan(filename="test-data-xrd/corundum.brml",
                       phase=corundum)
        df = scan.diffractogram
        q = df.index
        I = df.counts
        # Remove expected XRD peaks
        for reflection in corundum.reflection_list:
            q, I = remove_peak_from_df(x=q, y=I, xrange=reflection.qrange)
        # Do the background fitting
        bg = fit_background(q, I)
        assert False, "TODO: Write a test for making sure the bg fit is good"
