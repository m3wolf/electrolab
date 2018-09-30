# -*- coding: utf-8 -*-
#
# Copyright © 2018 Mark Wolf
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

import shutil
import unittest
import os
import sys


try:
    from scimap.gsas_refinement import GSASRefinement, import_gsas
    gsas = import_gsas(('~/build/pyGSAS/',))
except ImportError:
    has_gsas = False
else:
    has_gsas = True

TESTDIR = os.path.dirname(__file__)

# @unittest.skipUnless(has_gsas, 'GSAS-II not found')
@unittest.skip('GSAS-II refinement not ready yet.')
class GSASTestCase(unittest.TestCase):
    gpx_root = os.path.join(TESTDIR, 'refinement-temp', 'refinement0')
    gpx_template = os.path.join(TESTDIR, 'test-data-xrd', 'corundum-template.gpx')
    
    def remove_file(self, fname):
        if os.path.exists(fname):
            os.remove(fname)
    
    def setUp(self):
        shutil.copy(self.gpx_template, self.gpx_root + '.gpx')
    
    def tearDown(self):
        gpx_fname = self.gpx_root + '.gpx'
        self.remove_file(gpx_fname)
    
    def test_setup_gpx_file(self):
        refinement = GSASRefinement(file_root=self.gpx_root)
        gpx = refinement.gpx
        gpx_fname = os.path.abspath(self.gpx_root + '.gpx')
        self.assertEqual(gpx.filename, gpx_fname)
        # Check that phases are imported properly
        self.assertEqual(len(gpx.phases()), 1)
        # Make sure file isn't re-created when called
        refinement._gpx = 135
        self.assertEqual(refinement.gpx, 135)
    
    def test_add_data(self):
        refinement = GSASRefinement(file_root=self.gpx_root)
        gpx = refinement.gpx
        histos = gpx.histograms()
        histo = histos[1]
    
    @unittest.expectedFailure
    def test_refine(self):
        refinement = GSASRefinement(file_root=self.gpx_root,
                                    gpx_template=self.gpx_template)
        refinement.refine(self)
