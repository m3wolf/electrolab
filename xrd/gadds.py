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

# flake8: noqa

"""Series of functions for controlling a Bruker D8 Discover Series II
with GADDS controller software. Importing an already-acquired map is
done with the importers.import_gadds_map function."""

import os
import math

import numpy as np
import jinja2

import hdf
from default_units import angstrom
from mapping.coordinates import Cube
from .utilities import q_to_twotheta, twotheta_to_q
from .tube import tubes
from .xrdstore import XRDStore

# CHANGING THESE RISKS DAMAGE TO THE INSTRUMENT!
SOURCE_RANGE = (0, 50)
DETECTOR_RANGE = (0, 55)

def _source_angle(two_theta_range):
    """Compute the best source angle given the desired total diffraction
    angle.

    Arguments
    ---------
    - two_theta_range : Range of desired angles."""
    # Check for values outside preset limits
    theta1 = two_theta_range[0]
    if theta1 < SOURCE_RANGE[0]:
        msg = "2θ range {given} is outside source limits: {limits}"
        msg = msg.format(given=two_theta_range,
                         limits=SOURCE_RANGE)
        raise ValueError(msg)
    # Use either the desired value or the maximum source angle
    return min(theta1, SOURCE_RANGE[1])

def _detector_start(two_theta_range, frame_width):
    """Determine the best starting position for the detector. Assumes the
    source is at the highest angle reasonable.

    Arguments
    ---------
    - two_theta_range : 2-tuple with starting and ending angles in degrees.

    - frame_width : Usable integration range in degrees for each
      detector frame.
    """
    # Assuming that theta1 starts at highest possible range
    theta1 = _source_angle(two_theta_range=two_theta_range)
    theta2_bottom = two_theta_range[0] - theta1
    theta2_start = theta2_bottom + frame_width / 2
    return theta2_start

def _path(rows):
    """Generator gives coordinates for a spiral path around the sample."""
    loci = []
    # Six different directions one can move
    basis_set = {
        'W': Cube(-1, 1, 0),
        'SW': Cube(-1, 0, 1),
        'SE': Cube(0, -1, 1),
        'E': Cube(1, -1, 0),
        'NE': Cube(1, 0, -1),
        'NW': Cube(0, 1, -1)
    }
    # Start in the center
    curr_coords = Cube(0, 0, 0)
    loci.append(curr_coords)
    # Spiral through each row
    for row in range(1, rows):
        # Move to next row
        curr_coords += basis_set['NE']
        loci.append(curr_coords)
        for i in range(0, row - 1):
            curr_coords += basis_set['NW']
            loci.append(curr_coords)
        # Go around the ring for each basis vector
        for key in ['W', 'SW', 'SE', 'E', 'NE']:
            vector = basis_set[key]
            for i in range(0, row):
                curr_coords += vector
                loci.append(curr_coords)
    return loci

def _frame_step(detector_distance):
    if detector_distance == 20:
        step = 20
    else:
        msg = "Cannot determine step for detector distance {}".format(detector_distance)
        raise ValueError(msg)
    return step

def _context(diameter, collimator, coverage, scan_time,
             two_theta_range, detector_distance, frame_size, center,
             sample_name, hexadecimal):
    # Calculate some required values
    unit_size = math.sqrt(3) * collimator / 2
    frame_step = _frame_step(detector_distance=detector_distance)
    number_of_frames = _number_of_frames(two_theta_range, frame_step=frame_step)
    # (Unit size should be bigger if we're not mapping 100%)
    unit_size = unit_size / math.sqrt(coverage)
    # Prepare mapping path
    rows = math.ceil(diameter / unit_size / math.sqrt(3) + 1) # +1 is for center dot
    path = _path(rows)
    # Estimate the total time
    totalSecs = len(path) * scan_time * number_of_frames
    days = math.floor(totalSecs / 60 / 60 / 24)
    remainder = totalSecs - days * 60 * 60 * 24
    hours = math.floor(remainder / 60 / 60)
    remainder = remainder - hours * 60 * 60
    mins = math.floor(remainder / 60)
    total_time = "{secs}s ({days}d {hours}h {mins}m)".format(secs=totalSecs,
                                                             days=days,
                                                             hours=hours,
                                                             mins=mins)
    # List of frames to integrate
    frames = []
    for frame_num in range(0, number_of_frames):
        start = two_theta_range[0] + frame_num * frame_step
        end = start + frame_step
        frame = {
            'start': start,
            'end': end,
            'number': frame_num,
        }
        frames.append(frame)
    # Generate flood and spatial reference files to load
    floodFilename = "{framesize:04d}_{distance:03d}._FL".format(
        distance=round(detector_distance),
        framesize=frame_size
    )
    spatialFilename = "{framesize:04d}_{distance:03d}._ix".format(
        distance=round(detector_distance),
        framesize=frame_size
    )
    # Prepare context dictionary
    context = {
        'scans': [],
        'num_scans': len(path),
        'frames': frames,
        'frame_step': frame_step,
        'number_of_frames': number_of_frames,
        'xoffset': center[0],
        'yoffset': center[1],
        'theta1': _source_angle(two_theta_range=two_theta_range),
        'theta2': _detector_start(two_theta_range, frame_width=frame_step),
        'aux': 6,
        'scan_time': scan_time,
        'total_time': total_time,
        'sample_name': sample_name,
        'flood_file': floodFilename,
        'spatial_file': spatialFilename,
    }
    for idx, locus in enumerate(path):
        # Prepare scan-specific details
        x, y = locus.to_xy(unit_size=unit_size)
        if hexadecimal:
            map_template = "map-{n:x}"
        else:
            map_template = "map-{n:d}"
        scan_metadata = {'x': x, 'y': y, 'filename': map_template.format(n=idx)}
        context['scans'].append(scan_metadata)
    return context

def write_gadds_script(qrange, sample_name, center, file=None,
                       quiet=False, collimator=0.8, diameter=12.7,
                       coverage=1, scan_time=300, tube="Cu",
                       detector_distance=20, hexadecimal=False,
                       frame_size=1024, hdf_filename=None):
    wavelength = angstrom(tubes[tube].kalpha)
    two_theta_range = q_to_twotheta(np.array(qrange), wavelength=wavelength)
    # Import template
    env = jinja2.Environment(loader=jinja2.PackageLoader('scimap', ''))
    template = env.get_template('mapping/mapping-template.slm')
    context = _context(diameter=12.7, collimator=collimator,
                       coverage=1, scan_time=scan_time,
                       sample_name=sample_name,
                       two_theta_range=two_theta_range,
                       detector_distance=detector_distance,
                       center=center, hexadecimal=hexadecimal,
                       frame_size=frame_size)
    # Create file and directory if necessary
    if file is None:
        directory = '{}-frames'.format(sample_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = '{dir}/{samplename}.slm'.format(
            dir=directory, samplename=sample_name
        )
        with open(filename, 'w') as file:
            file.write(template.render(**context))
    else:
        file.write(template.render(**context))
    # Prepare HDF5 file to hold the results
    if hdf_filename is None:
        hdf_filename = "{}.h5".format(sample_name)
    hdfgroup = hdf.prepare_hdf_group(filename=hdf_filename,
                                     groupname=sample_name,
                                     dirname=None)
    xrdstore = XRDStore(hdf_filename=hdf_filename,
                        groupname=sample_name, mode="r+")
    positions = [(s['x'], s['y']) for s in context['scans']]
    xrdstore.positions = np.array(positions)
    xrdstore.layout = 'hex'
    file_basenames = np.array([s['filename'] for s in context['scans']])
    xrdstore.file_basenames = np.array(file_basenames)
    tube_ = tubes[tube]
    kalphas = (tube_.kalpha1, tube_.kalpha2)
    kalphas = np.array([k.num for k in kalphas])
    xrdstore.wavelength = kalphas
    xrdstore.group()['wavelength'].attrs['unit'] = "Å"
    hdfgroup.file.close()
    # Print summary info
    if not quiet:
        msg = "Running {num} scans ({frames} frames each). ETA: {time}."
        print(msg.format(num=context['num_scans'],
                         time=context['total_time'],
                         frames=context['number_of_frames']))
        frameStart = context['frames'][0]['start']
        frameEnd = context['frames'][-1]['end']
        msg = "Integration range: {start}° to {end}°"
        print(msg.format(start=frameStart, end=frameEnd))
    return file

def _number_of_frames(two_theta_range, frame_step):
    angle_range = two_theta_range[1] - two_theta_range[0]
    num_frames = math.ceil(angle_range / frame_step)
    # Check for values outside instrument limits
    t2_start = _detector_start(two_theta_range=two_theta_range,
                               frame_width=frame_step)
    t2_end = t2_start + num_frames * frame_step
    if (t2_end - SOURCE_RANGE[1]) > DETECTOR_RANGE[1]:
        msg = "2θ range {given} is outside detector limits: {limits}"
        msg = msg.format(given=two_theta_range,
                         limits=DETECTOR_RANGE)
        raise ValueError(msg)
    return num_frames
