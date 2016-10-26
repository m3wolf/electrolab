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
# along with Scimap. If not, see <http://www.gnu.org/licenses/>.

import numpy as np

from ..default_units import angstrom


def q_to_twotheta(q, wavelength):
    """Converts a numpy array or value in scattering length (q) into
    2θ angle.

    Arguments
    ---------
    - q : Number or numpy array of scattering lengths.

    - wavelength : Wavelength of the radiation. The value is assumed
    to be in angstroms unless it is a Quantity object.

    Returns
    -------
    Number or numpy array of two theta values in degrees.
    """
    wavelength = angstrom(wavelength).num
    inner = (q * wavelength / 4 / np.pi)
    twotheta = np.degrees(2 * np.arcsin(inner))
    return twotheta

def twotheta_to_q(two_theta, wavelength, degrees=True):
    """Converts a numpy array or value in 2θ angle into
    scattering length (q).

    Arguments
    ---------
    - two_theta : Number or numpy array of diffraction angles in degrees.

    - wavelength : Wavelength of the radiation. The value is assumed
    to be in angstroms unless it is a Quantity object.

    - degrees : If true, the value of two_theta is assumed to be in
      degrees, otherwise it is in radians.

    Returns
    -------
    Number or numpy array of scattering lengths in inverse angstroms.

    """
    wavelength = angstrom(wavelength).num
    # Convert to radians if necessary
    if degrees:
        radians = np.radians(two_theta)
    else:
        radians = two_theta
    # Convert radians to scattering lengths
    q = 4 * np.pi / wavelength * np.sin(radians / 2)
    return q
