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

import h5py
import numpy as np
import pandas as pd

import hdf

def import_aps_32IDE_map(directory: str, wavelength: int,
                         shape: Tuple[int, int], step_size: Union[float, int],
                         hdf_filename=None, hdf_groupname=None):
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

    """
    # Prepare HDF file
    sample_group = hdf.prepare_hdf_group(filename=hdf_filename,
                                         groupname=hdf_groupname,
                                         dirname=directory)
    sample_group.file.create_dataset('version', data=2)
    intensities = []
    qs = []
    angles = []

    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
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
        # Determine if we need to convert to q
        if '2-Theta Angle (Degrees)' in xunits:
            qs.append(4*np.pi/wavelength*np.sin(np.radians(csv.index/2)))
        intensities.append(csv.values)
    # Save to hdf file
    sample_group.create_dataset('scattering_lengths', data=qs)
    sample_group.create_dataset('intensities', data=intensities)
    return qs, intensities
