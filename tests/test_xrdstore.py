# -*- coding: utf-8 -*-
#
# Copyright © 2017 Mark Wolf
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

import unittest

from sympy.physics import units

from scimap.xrdstore import XRDStore


class XRDStoreTests(unittest.TestCase):

    def test_step_size(self):
        store = XRDStore(hdf_filename="test-data-xrd/xrd-map-gadds.h5",
                         groupname="xrd-map-gadds")
        # For now, just check that the value can be retrieved without
        # throwing an error
        step_size = store.step_size

    def test_step_unit(self):
        store = XRDStore(hdf_filename="test-data-xrd/xrd-map-gadds.h5",
                         groupname="xrd-map-gadds")
        store.step_unit = 'um'
        step_unit = store.step_unit
        self.assertEqual(step_unit, units.um)
