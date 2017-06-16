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

from sympy.physics import units
import h5py


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
    
    def replace_dataset(self, name, data, *args, **kwargs):
        """Wrapper for h5py.create_dataset that removes the existing dataset
        if it exists."""
        # Remove the existing dataset if possible
        try:
            del self.group()[name]
        except KeyError:
            pass
        # Perform the actual group creation
        self.group().create_dataset(name=name, data=data, *args, **kwargs)
    
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
    def positions(self):
        return self.group()['positions'].value
    
    @positions.setter
    def positions(self, value):
        self.replace_dataset('positions', data=value)
    
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
    
    @positions.setter
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
    def goodness(self):
        return self.group()['goodness_of_fit']
    
    @goodness.setter
    def goodness(self, value):
        self.replace_dataset('goodness_of_fit', data=value)
    
    @property
    def photo_filenames(self):
        return self.group()['photo_filenames']
    
    @photo_filenames.setter
    def photo_filenames(self, value):
        # Convert the ASCII
        value = [bytes(s, encoding="UTF-8") for s in value]
        self.replace_dataset('photo_filenames', data=value)
    
    @property
    def fits(self):
        return self.group()['fits'].value
    
    @fits.setter
    def fits(self, value):
        self.replace_dataset('fits', data=value)
    
    @property
    def cell_parameters(self):
        return self.group()['cell_parameters'].value
    
    @cell_parameters.setter
    def cell_parameters(self, value):
        self.replace_dataset('cell_parameters', data=value)
        group = self.group()
        group['cell_parameters'].attrs['order'] = "(scan, phase, (a, b, c, α, β, γ))"
    
    @property
    def phase_fractions(self):
        return self.group()['phase_fractions']
    
    @phase_fractions.setter
    def phase_fractions(self, value):
        self.replace_dataset('phase_fractions', data=value)
    
    @property
    def scale_factor(self):
        return self.group()['scale_factor']
    
    @scale_factor.setter
    def scale_factor(self, value):
        self.replace_dataset('scale_factor', data=value)
        
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
    def wavelengths(self):
        return self.group()['wavelengths'].value
    
    @wavelengths.setter
    def wavelengths(self, value):
        self.replace_dataset('wavelengths', data=value)
    
    @property
    def layout(self):
        return self.group()['positions'].attrs['layout']
    
    @layout.setter
    def layout(self, value):
        self.group()['positions'].attrs['layout'] = value
    
    @property
    def intensities(self):
        intensities = self.group()['intensities'].value
        return intensities
    
    @intensities.setter
    def intensities(self, value):
        self.replace_dataset('intensities', data=value)
    
    @property
    def intensities_subtracted(self):
        intensities_subtracted = self.group()['intensities_subtracted'].value
        return intensities_subtracted
        
    @intensities_subtracted.setter
    def intensities_subtracted(self, value):
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
    def scattering_lengths(self):
        q = self.group()['scattering_lengths'].value
        return q
    
    @scattering_lengths.setter
    def scattering_lengths(self, value):
        self.replace_dataset('scattering_lengths', data=value)
    
    @property
    def backgrounds(self):
        data = self.group()['backgrounds'].value
        new_shape = (data.shape[0], data.shape[1])
        return data.reshape(new_shape)
    
    @backgrounds.setter
    def backgrounds(self, value):
        name = 'backgrounds'
        try:
             del self.group()[name]
        except KeyError:
            pass
        self.group().create_dataset(name, data=value)
    
    @property
    def subtracted(self):
        bg = self.backgrounds
        it = self.intensities
        return it - bg
    
    @subtracted.setter
    def subtracted(self, value):
        name = 'subtracted'
        try:
            del self.group()[name]
        except KeyError:
            pass
        self.group().create_dataset(name, data=value)
    
    @property
    def peak_broadening(self):
        return self.group()['peak_broadening']
    
    @peak_broadening.setter
    def peak_broadening(self, value):
        self.replace_dataset('peak_broadening', data=value)
    
    @property
    def source(self):
        """Return a string indicating where the data came from."""
        return self.group().attrs['source']
