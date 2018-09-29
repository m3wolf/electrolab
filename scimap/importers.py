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

import os
import re
from typing import Union, Tuple
import warnings

import h5py
import numpy as np
import pandas as pd

from . import exceptions, hdf, utilities, default_units as units
from .utilities import twotheta_to_q, q_to_twotheta
from .adapters import BrukerPltFile
from .xrdstore import XRDStore


def import_gadds_map(sample_name: str=None, directory: str=None,
                     hdf_filename: str=None, hdf_groupname: str=None):
    """Import a set of diffraction patterns from a map taken on a Bruker
    D8 Discover Series II diffractometer using the GADDS software
    suite.
    
    Parameters
    ----------
    sample_name : optional
      Name describing this sample. If provided and not None, this can
      be used to guess the directory, hdf_filename and hdf_groupname
      arguments (otherwise they must be explicitely provided).
    directory : optional
      Directory where to look for results. It should contain .plt
      files that are 2-theta and intensity data as well as .jpg files
      of the locus images.
    hdf_filename : optional
      HDF File used to store computed results. If omitted or None, the
      `directory` basename is used
    hdf_groupname : optional
      String to use for the hdf group of this dataset. If omitted or
      None, the `directory` basename is used. Raises an exception if
      the group already exists in the HDF file.
    
    """
    # Check that the we have all the right filenames provided in arguments
    if sample_name is None and None in [hdf_filename, hdf_groupname, directory]:
        msg = "Either pass `sample_name` or `hdf_filename`, `hdf_groupname` and `directory`"
        raise ValueError(msg)
    if hdf_filename is None:
        hdf_filename = "{}.h5".format(sample_name)
    if hdf_groupname is None:
        hdf_groupname = sample_name
    if directory is None:
        directory = "{}-frames/".format(sample_name)
    # Make sure the HDF file exists
    if not os.path.exists(hdf_filename):
        msg = "HDF file '{}' not found. Did you remember to run `write_gadds_script`?"
        raise OSError(msg.format(hdf_filename))
    # Open HDF datastore
    xrdstore = XRDStore(hdf_filename=hdf_filename,
                        groupname=hdf_groupname, mode="r+")
    xrdstore.group().attrs['source'] = "gadds"
    wavelength = xrdstore.effective_wavelength
    # Prepare list of .plt and .jpg files
    basenames = xrdstore.file_basenames
    filestring = os.path.join(directory, "{base}.{ext}")
    pltfiles = [filestring.format(base=base.decode(), ext="plt") for base in basenames]
    jpgfiles = [os.path.join(directory, str(base) + ".jpg") for base in basenames]
    # Arrays to hold imported results
    Is, two_thetas = [], []
    # Read plt data files
    for filename in pltfiles:
        try:
            plt = BrukerPltFile(filename=filename)
            Is.append(plt.intensities())
            two_thetas.append(plt.two_theta())
        except FileNotFoundError:
            raise exceptions.MappingFileNotFoundError(
                'Cannot find file `{}`'.format(filename))
    # Save diffraction data to HDF5 file
    Is = np.array(Is)
    xrdstore.intensities = Is
    two_thetas = np.array(two_thetas)
    xrdstore.two_thetas = two_thetas
    xrdstore.photo_filenames = jpgfiles
    # Clean up
    xrdstore.close()


def import_aps_34IDE_map(directory: str, wavelength: int,
                         shape: Tuple[int, int], step_size: Union[float, int],
                         hdf_filename=None, hdf_groupname=None,
                         beamstop=0, qrange=None):
    """Import a set of diffraction patterns from a map taken at APS
    beamline 34-ID-E. The data should be taken in a rectangle.
    
    Arguments
    ---------
    
    directory
      Directory where to look for results. It should contain .chi
      files that are q or 2-theta and intensity data."
    wavelength
      Wavelength of x-ray used, in angstroms.
    shape
      2-tuple for number of scanning loci in each direction. The first
      value is the slow axis and the second is the fast axis.
    step_size
      Number indicating how far away each locus is from every
      other. Best practice is to include units directly by using the
      `units` package.
    hdf_filename : optional
      HDF File used to store computed results. If omitted or None, the
      `directory` basename is used
    hdf_groupname : optional
      String to use for the hdf group of this dataset. If omitted or
      None, the `directory` basename is used. Raises an exception if
      the group already exists in the HDF file.
    beamstop : deprecated
      A scattering length (q) below which the beam stop cuts off the
      signal. Use ``qrange`` instead.
    qrange : optional
      A scattering length (q) range beyond the signal is cut
      invalid. This helps remove things like beam stop effects.
    
    """
    if hdf_filename is None:
        # Set a default filename based on the directory name
        dirnames = directory.split('/')
        dirnames.remove('')
        hdf_filename = dirnames[-1] + ".h5"
    # Prepare HDF file
    sample_group = hdf.prepare_hdf_group(filename=hdf_filename,
                                         groupname=hdf_groupname,
                                         dirname=directory)
    xrdstore = XRDStore(hdf_filename=hdf_filename, groupname=sample_group.name, mode="r+")
    xrdstore.group().attrs['source'] = "APS 34-ID-E"
    wavelength_AA = [(wavelength, 1)]
    sample_group.create_dataset('wavelengths', data=wavelength_AA)
    sample_group['wavelengths'].attrs['unit'] = 'Å'
    # Determine the sample step sizes
    xrdstore.step_size = step_size
    # Calculate mapping positions ("loci")
    shape = utilities.shape(*shape)
    xs = np.arange(0, shape.columns*step_size, step_size)
    ys = np.arange(0, shape.rows*step_size, step_size)
    xv, yv = np.meshgrid(xs, ys)
    newshape = (shape.rows * shape.columns, 2)
    positions = np.reshape(np.dstack((xv, yv)), newshape=newshape)
    # Shift positions by half a step-size so the location is the center of each square
    positions = np.add(positions, step_size / 2)
    xrdstore.positions = positions
    xrdstore.position_unit = 'um'
    xrdstore.layout = 'rect'
    intensities = []
    two_thetas = []
    angles = []
    file_basenames = []
    chifiles = [p for p in os.listdir(directory) if os.path.splitext(p)[1] == '.chi']
    if beamstop:
        warnings.warn(UserWarning("Deprecated, use qrange instead"))
    # Sort filenames
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    for filename in sorted(chifiles, key=alphanum_key):
        path = os.path.join(directory, filename)
        file_basenames.append(os.path.splitext(path)[0])
        # Get header data
        with open(path) as f:
            lines = f.readlines()
            xunits = lines[1].strip()
            yunits = lines[2].strip()
            num_points = int(lines[3].strip())
        # Load diffraction pattern
        csv = pd.read_table(path,
                          sep='\s+',
                          header=0,
                          index_col=0,
                          names=[xunits, yunits],
                          skiprows=4,
                          skipinitialspace=True)
        # Determine if we need to convert to q (scattering vector length)
        if '2-Theta Angle (Degrees)' in xunits:
            # Remove values obscured by the beam stop
            if beamstop > 0:
                warnings.warn(UserWarning("Deprecated, use qrange instead"))
                csv = csv.loc[q_to_twotheta(beamstop, wavelength=wavelength_AA):]
            elif qrange is not None:
                angle_range = [q_to_twotheta(q, wavelength=wavelength_AA) for q in qrange]
                csv = csv.loc[angle_range[0]:angle_range[1]]
            # Convert to scattering factor
            q = twotheta_to_q(csv.index, wavelength=wavelength_AA)
        elif 'Q (Inverse Nanometres)' in xunits:
            # Convert from inverse nanometers to inverse angstroms
            if qrange is not None:
                _qrange = tuple(_q * float(units.nm / units.angstrom) for _q in qrange)
                csv = csv.loc[_qrange[0]:_qrange[1]]
            q = csv.index
            q = q * float(units.angstrom / units.nm)
        else: # Data in unknown format
            raise exceptions.FileFormatError("Cannot recognize {}".format(xunits))
            # Remove values obscured by the beam stop
        qs.append(q)
        intensities.append(csv.values)
    # Convert to properly shaped numpy arrays
    qs = np.array(qs)
    intensities = np.array(intensities)
    new_shape = (intensities.shape[0], intensities.shape[1])
    intensities = intensities.reshape(new_shape)
    file_basenames = np.array(file_basenames).astype("S30")
    # Save to hdf file
    sample_group.create_dataset('scattering_lengths', data=qs)
    sample_group['scattering_lengths'].attrs['unit'] = 'Å⁻'
    sample_group.create_dataset('intensities', data=intensities)
    sample_group.create_dataset('file_basenames', data=file_basenames)
    # return qs, intensities
