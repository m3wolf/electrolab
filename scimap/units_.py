# -*- coding: utf-8 -*-
#
# Copyright © 2018 Mark Wolfman
#
# This file is part of Ccimap.
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

"""Holds unit definitions specific to mapping, including default units
for certain properties."""

from pint import UnitRegistry


class DefaultUnit():
    """Descriptor for accessing default units from the registry."""
    def __init__(self, name):
        self.name = name
    
    def __get__(self, obj, objtype):
        print(obj)
        return getattr(obj, self.name)


class ScimapUnits(UnitRegistry):
    # Add handlers for default units to the registry
    wavelength = DefaultUnit('angstrom')
    mass = DefaultUnit('gram')


units = ScimapUnits()
