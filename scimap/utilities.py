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

"""A collection of classes and functions that arent' specific to any
one type of measurement. Also, it defines some namedtuples for
describing coordinates.
"""

from collections import namedtuple
import math
import sys

from tqdm import tqdm
import numpy as np
from sympy.physics import units

xycoord = namedtuple('xycoord', ('x', 'y'))
Pixel = namedtuple('Pixel', ('vertical', 'horizontal'))
shape = namedtuple('shape', ('rows', 'columns'))


def q_to_twotheta(q, wavelength):
    """Converts a numpy array or value in scattering length (q) into
    2θ angle.

    Parameters
    ----------
    q
      Number or numpy array of scattering lengths.
    wavelength
      Wavelength of the radiation. This parameter can (and should) use
      ``sympy.physics.units``

    Returns
    -------
    Number or numpy array of two theta values in degrees.

    """
    inner = (q * wavelength / 4 / np.pi)
    try:
        # If it's a numpy array, convert to a float
        inner = inner.astype(np.float)
    except AttributeError as e:
        # Probably a float or integer
        inner = float(inner)
    except SystemError as e:
        raise SystemError("Did you remember to put units on q?", e)
    twotheta = np.degrees(2 * np.arcsin(inner))
    return twotheta


def twotheta_to_q(two_theta, wavelength, degrees=True):
    """Converts a numpy array or value in 2θ angle into
    scattering length (q). 
    
    Parameters
    ----------
    two_theta
      Number or numpy array of diffraction angles in degrees.
    wavelength
      Wavelength of the radiation. It can (and probably should) have a
      unit from ``sympy.physics.units``
    degrees
      If true, the value of two_theta is assumed to be in
      degrees, otherwise it is in radians.
    
    Returns
    -------
    Number or numpy array of scattering lengths in inverse angstroms.
    
    """
    # Convert to radians if necessary
    if degrees:
        radians = np.radians(two_theta)
    else:
        radians = two_theta
    # Convert radians to scattering lengths
    q = 4 * np.pi / wavelength * np.sin(radians / 2)
    return q


class Prog:
    """A progress bar for displaying how many iterations have been
    completed. This is mostly just a wrapper around the tqdm
    library. Additionally it makes use of the borg pattern, so setting
    Prog.quiet to True once silences all progress bars. This is useful
    for testing.
    """
    __global_state = {
        'quiet': False
    }

    def __init__(self):
        self.__dict__ = self.__global_state

    def __call__(self, iterable, *args, **kwargs):
        """Progress meter. Wraps around tqdm with some custom defaults."""
        if self.quiet:
            # Supress outputs by redirecting to stdout
            # kwargs['file'] = open(os.devnull, 'w')
            # # kwargs['file'] = os.devnull
            ret = iterable
        else:
            kwargs['file'] = kwargs.get('file', sys.stdout)
            kwargs['leave'] = kwargs.get('leave', True)
            ret = tqdm(iterable, *args, **kwargs)
        return ret

prog = Prog()
