# -*- coding: utf-8 -*-
#
# Copyright © 2016 Mark Wolf
#
# This file is part of Scimap.
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

"""Function and classes that prepare experiments at specific
synchrotron beamlines."""

from typing import List, Tuple, Iterable
from collections import namedtuple

from .edges import Edge
from .frame import position

ZoneplatePoint = namedtuple('ZoneplatePoint', ('z', 'energy'))
DetectorPoint = namedtuple('DetectorPoint', ('z', 'energy'))


class Zoneplate():
    """Type of focusing optic using in X-ray microscopy. It must be moved
    with changing energy to properly focus the beam. In order to
    properly predict zoneplate positions, it needs either two
    position-energy pairs or one position-energy pair and a step.

    Arguments
    ---------
    - start : The first zoneplate position-energy pair.

    - step : Adjustment in z-position for every positive change of 1 eV
      of beam energy.

    - end : The second zoneplate position-energy pair.
    """
    def __init__(self,
                 start: ZoneplatePoint,
                 step: int=None,
                 end: ZoneplatePoint=None):
        # Check sanity of arguments
        if step is None and end is None:
            msg = "Either `step` or `end` is required."
            raise ValueError(msg)
        elif step is not None and end is not None:
            msg = "Passing both `step` or `end` is confusing."
            raise ValueError(msg)
        elif step is None:
            # Calculate the step from start and end points
            self.step = (end.z - start.z) / (end.energy - start.energy)
        else:
            self.step = step
        self.start = start

    def z_position(self, energy: float):
        """Predict the z-position of the zoneplate for the given energy."""
        new_z = self.start.z + self.step * (energy - self.start.energy)
        return new_z


class Detector(Zoneplate):
    """A calibration object for the position of the detector."""
    pass


def sector8_xanes_script(dest,
                         edge: Edge,
                         zoneplate: Zoneplate,
                         detector: Detector,
                         sample_positions: List[position],
                         names: List[str],
                         iterations: Iterable=range(0, 1),
                         binning: int=1,
                         exposure: int=30):
    """Prepare an script file for running multiple consecutive XANES
    framesets on the transmission x-ray micrscope at the Advanced
    Photon Source beamline 8-BM-B.

    Arguments
    ---------
    - dest : file-like object that will hold the resulting script

    - edge : Description of the absorption edge

    - binning : how many CCD pixels to combine into one image pixel
      (eg. 2 means 2x2 CCD pixels become 1 image pixel.

    - exposure : How many seconds to collect for per frame

    - sample_positions : Locations to move the x, y (and z) axes to in
      order to capture the image.

    - zoneplate : Calibration details for the Fresnel zone-plate.

    - detector : Like zoneplate, but for detector.

    - names : sample name to use in file names.

    - iterations : iterable to contains an identifier for each full
      set of xanes location with reference.

    """
    dest.write("setbinning {}\n".format(binning))
    dest.write("setexp {}\n".format(exposure))
    energies = edge.all_energies()
    starting_energy = energies[0]
    for iteration in iterations:
        for idx in range(0, len(sample_positions)):
            position = sample_positions[idx]
            name = names[idx]
            # Move to x, y, z
            dest.write("moveto x {:.2f}\n".format(position.x))
            dest.write("moveto y {:.2f}\n".format(position.y))
            dest.write("moveto z {:.2f}\n".format(position.z))
            # Approach target energy from below
            for energy in range(starting_energy - 100, starting_energy, 2):
                dest.write("moveto energy {:.2f}\n".format(energy))
            for energy in energies:
                # Set energy
                dest.write("moveto energy {:.2f}\n".format(energy))
                # Set zoneplate
                dest.write("moveto zpz {:.2f}\n".format(zoneplate.z_position(energy)))
                # Set detector
                dest.write("moveto detz {:.2f}\n".format(detector.z_position(energy)))
                # Collect frame
                filename = "{name}_xanes{iter}_{energy}eV.xrm"
                energy_str = "{}_{}".format(*str(float(energy)).split('.'))
                filename = filename.format(name=name,
                                           iter=iteration,
                                           energy=energy_str)
                dest.write("collect {filename}\n".format(filename=filename))
