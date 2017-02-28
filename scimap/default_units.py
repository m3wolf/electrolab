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

"""Define common units across the whole application. Dataframes
assume the following units and can convert after calculation. This
module star-imports everything from sympy.physics.units.

"""

from sympy.physics.units import *

# Define default units

mass = gram

# time = hour

# capacity = milli * ampere * hour

# specific_capacity = milli * ampere * hour / gram

# potential = volts

# electrode_loading = mg / (cm * cm)

# energy = eV

angstrom = meter * (1e-10)
# wavelength = angstrom
