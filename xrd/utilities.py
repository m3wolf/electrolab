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


def q_to_twotheta(q, wavelength):
    """Converts a numpy array or value in scattering length (q) into
    2θ angle."""
    twotheta = np.degrees(2 * np.arcsin(q * wavelength / 4 / np.pi))
    return twotheta

def twotheta_to_q(two_theta, wavelength):
    """Converts a numpy array or value in 2θ angle into scattering length
    (q)."""
    q = 4 * np.pi / wavelength * np.sin(np.radians(two_theta / 2))
    return q
