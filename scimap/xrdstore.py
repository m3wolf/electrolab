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
from scimap import exceptions
import numpy as np

from . import exceptions
from .units_ import units


class StoreDescriptor():
    """Data descriptor for accessing HDF datasets.
    
    Parameters
    ----------
    name : str
      The dataset name in the HDF file.
    context : str, optional
      Type of dataset this is: frameset, map, metadata, etc.
    dtype : np.dtype, optional
      The data-type to use when saving new data to disk. Using lower
      precision datatypes can save significant disk space.
    
    """
    def __init__(self, name, dtype=None):
        self.name = name
        self.dtype = dtype
    
    def __get__(self, store, type=None):
        dataset = store.get_dataset(self.name)
        return dataset
    
    def __set__(self, store, value):
        store.replace_dataset(name=self.name, data=value, dtype=self.dtype)
    
    def __delete__(self, store):
        del store.data_group()[self.name]


class XRDStore():
    """Wrapper around HDF file that stores XRD data.
    
    Parameters
    ----------
    hdf_filename : str
      Filename for the HDF5 file to use for this store.
    groupname : str
      Top-level groupname to use for this sample. Multiple samples can
      be kept in one HDF5 file by assigning different group names.
    
    """
    VERSION = 1
    # Data descriptors that pull from HDF5 file
    positions = StoreDescriptor('positions')
    goodness_of_fit = StoreDescriptor('goodness_of_fit')
    intensities = StoreDescriptor('intensities')
    phase_fractions = StoreDescriptor('phase_fractions')
    scale_factor = StoreDescriptor('scale_factor')
    fits = StoreDescriptor('fits')
    wavelengths = StoreDescriptor('wavelengths')
    # scattering_lengths = StoreDescriptor('scattering_lengths')
    two_thetas = StoreDescriptor('two_thetas')
    backgrounds = StoreDescriptor('backgrounds')
    peak_broadenings = StoreDescriptor('peak_broadenings')

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
    
    def group(self):
        return self._file[self.groupname]
    
    def get_dataset(self, name):
        return self.group()[name]
    
    def replace_dataset(self, name, data, *args, **kwargs):
        """Wrapper for h5py.create_dataset that replaces the existing dataset.
        
        HDF5 attributes are copied to the new dataset.
        
        """
        # Remove the existing dataset if possible
        try:
            attrs = self.group()[name].attrs
            del self.group()[name]
        except KeyError:
            attrs = {}
        # Perform the actual group creation
        new_ds = self.group().create_dataset(name=name, data=data, *args, **kwargs)
        new_ds.attrs.update(attrs)
    
    @property
    def step_size(self):
        unit_name = self.group()['step_size'].attrs['unit']
        unit = getattr(units, unit_name)
        val = self.group()['step_size'].value
        return val * unit
    
    @step_size.setter
    def step_size(self, value):
        # Check if the given value is a composed unit
        if hasattr(value, 'unit'):
            step_size_value = value.num
            step_size_unit = value.unit.name
        else:
            step_size_value = value
            step_size_unit = 'mm'
        # Save values to HDF5 file
        self.replace_dataset('step_size', data=step_size_value)
        self.group()['step_size'].attrs['unit'] = step_size_unit
    
    @property
    def position_unit(self):
        unit = self.group()['positions'].attrs['unit']
        unit = getattr(units, unit)
        return unit
    
    @position_unit.setter
    def position_unit(self, val):
        self.group()['positions'].attrs['unit'] = val
    
    @property
    def layout(self):
        return self.group()['positions'].attrs['layout']
    
    @layout.setter
    def layout(self, value):
        self.group()['positions'].attrs['layout'] = value
    
    @property
    def file_basenames(self):
        return self.group()['file_basenames'].value
    
    @file_basenames.setter
    def file_basenames(self, value):
        value = value.astype("S10")
        self.replace_dataset('file_basenames', data=value)
    
    @property
    def photo_filenames(self):
        return self.group()['photo_filenames']
    
    @photo_filenames.setter
    def photo_filenames(self, value):
        # Convert the ASCII
        value = [bytes(s, encoding="UTF-8") for s in value]
        self.replace_dataset('photo_filenames', data=value)
    
    @property
    def cell_parameters(self):
        return self.group()['cell_parameters'].value
    
    @cell_parameters.setter
    def cell_parameters(self, value):
        self.replace_dataset('cell_parameters', data=value)
        group = self.group()
        group['cell_parameters'].attrs['order'] = "(scan, phase, (a, b, c, α, β, γ))"
    
    @property
    def effective_wavelength(self):
        wavelengths = self.wavelengths
        # Combine kα1 and kα2
        if len(wavelengths) == 2:
            wl = (wavelengths[0] + 0.5*wavelengths[1]) / 1.5
        else:
            wl = wavelengths
        return wl
    
    @property
    def layout(self):
        return self.group()['positions'].attrs['layout']
    
    @layout.setter
    def layout(self, value):
        self.group()['positions'].attrs['layout'] = value
    
    @property
    def intensities_subtracted(self):
        raise exceptions.DeprecationError("Just subtract them in real time")
        intensities_subtracted = self.group()['intensities_subtracted'].value
        return intensities_subtracted
    
    @intensities_subtracted.setter
    def intensities_subtracted(self, value):
        raise exceptions.DeprecationError("Just subtract them in real time")
        self.replace_dataset('intensities_subtracted', data=value)
    
    @property
    def collimator(self):
        collimator = self.group()['collimator'].value
        return collimator
    
    @collimator.setter
    def collimator(self, value):
        self.replace_dataset('collimator', data=value)
        self.group()['collimator'].attrs['unit'] = 'mm'
    
    @property
    def subtracted(self):
        raise exceptions.DeprecationError("Just subtract them as needed.")
        bg = self.backgrounds
        it = self.intensities
        return it - bg
    
    @property
    def source(self):
        """Return a string indicating where the data came from."""
        return self.group().attrs['source']
