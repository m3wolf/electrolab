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

from scimap import XRDScan, standards, Phase, Reflection



class PhaseTest(unittest.TestCase):
    class TestPhase(Phase):
        reflection_list = [
            Reflection('111'),
            Reflection('130'),
        ]
    
    def test_reflection_list(self):
        """Check that reflections get reset during initalization. This is
        necessary so that multiple phases can be manipulated and
        refined simultaneously.
        
        """
        phase0 = self.TestPhase()
        ref0 = phase0.reflection_list[0]
        phase1 = self.TestPhase()
        ref1 = phase1.reflection_list[0]
        # Change a reflection and ensure that the other one stays the same
        ref0.intensity = 77
        self.assertIsNot(ref0, ref1)
        self.assertNotEqual(ref1.intensity, 77)

