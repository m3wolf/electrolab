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
# along with scimap.  If not, see <http://www.gnu.org/licenses/>.

# flake8: noqa

import sys
import os

# Make sure this directory is in python path for imports
# sys.path.append(os.path.dirname(__file__))

from .peakfitting import Peak, remove_peak_from_df
# from xrd.peak import XRDPeak
# from xrdpeak import PeakFit

from . import standards, lmo, nca, gadds, fullprof_refinement, filters
from .peakfitting import Peak
from .unitcell import CubicUnitCell, HexagonalUnitCell, TetragonalUnitCell
from .reflection import Reflection
from .tube import tubes
from .phase import Phase
from .fullprof_refinement import FullProfPhase, FullprofRefinement
from .peak import XRDPeak
from .scan import XRDScan, align_scans
from .gadds import write_gadds_script
from .importers import import_aps_34IDE_map, import_gadds_map
from .utilities import q_to_twotheta, twotheta_to_q, xycoord, Pixel, shape, prog
from .plots import (new_axes, big_axes, dual_axes, plot_scans, xrd_axes,
                    plot_txm_intermediates, new_image_axes)
from .coordinates import Cube
from .xrd_map import XRDMap, Map

# Load units from unit registry
from .units_ import units
