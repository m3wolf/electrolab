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
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

# flake8: noqa

import sys
import os

# Make sure this directory is in python path for imports
sys.path.append(os.path.dirname(__file__))

import default_units
from electrochem.electrochem_units import *

from peakfitting import Peak
# from xrd.peak import XRDPeak
# from xrdpeak import PeakFit

from plots import (new_axes, big_axes, dual_axes, plot_scans, xrd_axes,
                   plot_txm_intermediates, new_image_axes)

import filters

from utilities import xycoord, Pixel, shape

from peakfitting import Peak

from refinement import fullprof

from xrd.unitcell import CubicUnitCell, HexagonalUnitCell, TetragonalUnitCell
from xrd import standards, lmo
from xrd.lmo import LMOPlateauMap
from xrd.reflection import Reflection
from xrd.peak import XRDPeak
from xrd.scan import XRDScan, align_scans
from xrd.map import XRDMap

from mapping.coordinates import Cube
from mapping.map import Map, DummyMap, PeakPositionMap, PhaseRatioMap, FwhmMap
from mapping import colormaps

# Electrochemistry methods and classes
from electrochem.electrode import CathodeLaminate, CoinCellElectrode
from electrochem.galvanostatrun import GalvanostatRun
from electrochem.cycle import Cycle
from electrochem.plots import plot_rate_capacities

# X-ray microscopy
from txm.xradia import XRMFile
from txm.beamlines import Zoneplate, Detector, sector8_xanes_script, ZoneplatePoint
from txm.importers import import_txm_framesets
from txm.xanes_frameset import XanesFrameset
from txm.frame import TXMFrame, calculate_particle_labels, rebin_image, position
from txm.edges import k_edges
from txm.plotter import FramesetMoviePlotter
