# -*- coding: utf-8 -*-
"""Sample definitions for nickel-cobalet-aluminum oxide
LiNi_{0.8}Co_{0.15}Al_{0.05}O_2

"""

from matplotlib.colors import Normalize

from .map import XRDMap
from .phase import Phase
from .standards import Aluminum
from .unitcell import CubicUnitCell, TetragonalUnitCell, HexagonalUnitCell
from .reflection import Reflection
from refinement.fullprof import FullProfPhase, ProfileMatch


class NCA(Phase):
    name = 'NCA'
    unit_cell = HexagonalUnitCell(a=2.86687, c=14.18385)
    spacegroup = 'R-3m'
    fullprof_spacegroup = 'R -3 M'
    diagnostic_hkl = '113'
    reflection_list = [
        Reflection('003', qrange=(1.30, 1.35)),
        Reflection('101', qrange=(2.50, 2.60)),
        Reflection('006', qrange=(2.60, 2.70)),
        Reflection('102', qrange=(2.62, 2.72)),
        Reflection('104', qrange=(3.05, 3.10)),
        Reflection('105', qrange=(3.30, 3.40)),
        Reflection('107', qrange=(3.95, 4.05)),
        Reflection('107', qrange=(3.95, 4.05)),
        Reflection('108', qrange=(4.30, 4.35)),
        Reflection('110', qrange=(4.35, 4.40)),
        Reflection('113', qrange=(4.50, 4.60)),
    ]

# Define a new class for mapping the transition
class NCAMap(XRDMap):
    scan_time = 300
    two_theta_range = (53, 73)
    phases = [NCA]
    background_phases = [Aluminum]
    # phase_ratio_normalizer = Normalize(0, 0.7, clip=True)
    # reliability_normalizer = Normalize(0.7, 2, clip=True)
