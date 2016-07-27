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
from typing import Union, Tuple

import h5py
import numpy as np
import pandas as pd

import hdf
from default_units import angstrom
from .adapters import BrukerPltFile
from .xrdstore import XRDStore
from .utilities import twotheta_to_q, q_to_twotheta


def import_gadds_map(directory: str, tube: str="Cu",
                     hdf_filename: str=None, hdf_groupname: str=None):
    """Import a set of diffraction patterns from a map taken on a Bruker
    D8 Discover Series II diffractometer using the GADDS software
    suite.

    Arguments
    ---------

    - directory : Directory where to look for results. It should
    contain .plt files that are 2-theta and intensity data as well as
    .jpg files of the locus images

    - tube : Anode material used in the X-ray tube. This will be used
      to determine the wavelength for converting two-theta to
      scattering lengths (q).

    - hdf_filename : HDF File used to store computed results. If
      omitted or None, the `directory` basename is used

    - hdf_groupname : String to use for the hdf group of this
      dataset. If omitted or None, the `directory` basename is
      used. Raises an exception if the group already exists in the HDF
      file.

    """
    # Open HDF datastore
    xrdstore = XRDStore(hdf_filename=hdf_filename,
                        groupname=hdf_groupname, mode="r+")
    wavelength = xrdstore.effective_wavelength
    # Prepare list of .plt and .jpg files
    basenames = xrdstore.file_basenames
    filestring = os.path.join(directory, "{base}.{ext}")
    pltfiles = [filestring.format(base=base.decode(), ext="plt") for base in basenames]
    jpgfiles = [os.path.join(directory, str(base) + ".jpg") for base in basenames]
    # Arrays to hold imported results
    Is, qs = [], []
    # Read plt data files
    for filename in pltfiles:
        plt = BrukerPltFile(filename=filename)
        Is.append(plt.intensities())
        qs.append(plt.scattering_lengths(wavelength=wavelength))
    # Save diffraction data to HDF5 file
    Is = np.array(Is)
    xrdstore.intensities = Is
    qs = np.array(qs)
    xrdstore.scattering_lengths = qs
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

    - directory : Directory where to look for results. It should
    contain .chi files that are q or 2-theta and intensity data."

    - wavelength : Wavelength of x-ray used, in angstroms.

    - shape : 2-tuple for number of scanning loci in each
      direction. The first value is the slow axis and the second is
      the fast axis.

    - step_size : Number indicating how far away each locus is from
      every other. Best practice is to include units directly by using
      the `units` package.

    - hdf_filename : HDF File used to store computed results. If
      omitted or None, the `directory` basename is used

    - hdf_groupname : String to use for the hdf group of this
      dataset. If omitted or None, the `directory` basename is
      used. Raises an exception if the group already exists in the HDF
      file.

    - beamstop : [deprecated] A scattering length (q) below which the beam stop
      cuts off the signal.

    - qrange : A scattering length (q) range beyond the signal is cut
      invalid. This helps remove things like beam stop effects.

    """
    # Prepare HDF file
    sample_group = hdf.prepare_hdf_group(filename=hdf_filename,
                                         groupname=hdf_groupname,
                                         dirname=directory)
    xrdstore = XRDStore(hdf_filename=hdf_filename, groupname=sample_group.name, mode="r+")
    wavelength_AA = angstrom(wavelength).num
    sample_group.create_dataset('wavelengths', data=wavelength_AA)
    sample_group['wavelengths'].attrs['unit'] = 'Å'
    # Determine the sample step sizes
    xrdstore.step_size = step_size
    # Calculate mapping positions ("loci")
    xv, yv = np.meshgrid(range(0, shape.columns), range(0, shape.rows))
    newshape = (shape.rows * shape.columns, 2)
    positions = np.reshape(np.dstack((xv, yv)), newshape=newshape)
    # Shift positions by half a step-size so the location is the center of each square
    positions = np.add(positions, step_size.num / 2)
    xrdstore.positions = positions
    xrdstore.layout = 'rect'

    intensities = []
    qs = []
    angles = []
    file_basenames = []

    chifiles = [p for p in os.listdir(directory) if os.path.splitext(p)[1] == '.chi']

    if beamstop:
        warnings.warn(UserWarning("Deprecated, use qrange instead"))

    for filename in sorted(chifiles):
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
        else:
            # Data already in q
            # Remove values obscured by the beam stop
            if beamstop > 0:
                csv = csv.loc[q_to_twotheta(beamstop):]
            q = csv.index
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
    return qs, intensities
