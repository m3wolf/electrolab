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

import h5py
import units


class XRDStore():
    """Wrapper around HDF file that stores XRD data.

    Arguments
    ---------

    - hdf_filename : Filename for the HDF5 file to use for this store.

    - groupname : Top-level groupname to use for this sample. Multiple
      samples can be kept in one HDF5 file by assigning different
      group names.

    """
    VERSION = 1
    def __init__(self, hdf_filename: str, groupname: str, mode='r'):
        self.hdf_filename = hdf_filename
        self.groupname = groupname
        self.mode = mode
        self._file = h5py.File(self.hdf_filename, mode=mode)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self._file.close()

    def _group(self):
        return self._file[self.groupname]

    @property
    def step_size(self):
        unit = units.unit(self._group()['step_size'].attrs['unit'])
        val = self._group()['step_size'].value
        return unit(val)

    @step_size.setter
    def step_size(self, value):
        # Check if the given value is a composed unit
        if hasattr(value, 'unit'):
            step_size_value = value.num
            step_size_unit = value.unit.name
        else:
            step_size_value = value
            step_size_unit = 'm'
        # Save values to HDF5 file
        self._group().create_dataset('step_size', data=step_size_value)
        self._group()['step_size'].attrs['unit'] = step_size_unit

    @property
    def positions(self):
        return self._group()['positions'].value

    @positions.setter
    def positions(self, value):
        self._group().create_dataset('positions', data=value)

    @property
    def layout(self):
        return self._group()['positions'].attrs['layout']

    @layout.setter
    def layout(self, value):
        self._group()['positions'].attrs['layout'] = value
