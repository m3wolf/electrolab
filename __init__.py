# -*- coding: utf-8 -*-

# Make sure this directory is in python path for imports
import sys
import os
sys.path.append(os.path.dirname(__file__))

import default_units
from electrochem.electrochem_units import *

from xrd.peak import XRDPeak
# from xrdpeak import PeakFit

from plots import new_axes, big_axes, dual_axes, plot_scans

import filters

from refinement import fullprof

from xrd.unitcell import CubicUnitCell, HexagonalUnitCell, TetragonalUnitCell
from xrd import standards, lmo
from xrd.lmo import LMOPlateauMap
from xrd.reflection import Reflection
from xrd.scan import XRDScan, align_scans
from xrd.map import XRDMap

from mapping.coordinates import Cube
from mapping.map import Map, DummyMap, PeakPositionMap, PhaseRatioMap, FwhmMap

# Electrochemistry methods and classes
from electrochem.electrode import CathodeLaminate, CoinCellElectrode
from electrochem.galvanostatrun import GalvanostatRun
from electrochem.cycle import Cycle
from electrochem.plots import plot_rate_capacities
