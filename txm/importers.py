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
import re
from time import time

from tqdm import format_meter
import pandas as pd
from PIL import Image

from .xradia import XRMFile
from .xanes_frameset import XanesFrameset, energy_key
from .frame import TXMFrame, average_frames
from utilities import prog
import exceptions

def _prepare_hdf_group(filename: str, groupname: str, dirname: str):
    """Check the filenames and create an hdf file as need. Throws an
    exception if the file already exists.

    Returns: HDFFile

    Arguments
    ---------

    - filename : name of the requested hdf file, may be None if not
      provided, in which case the filename will be generated
      automatically based on `dirname`.
    """
    # Get default filename and groupname if necessary
    if filename is None:
        real_name = os.path.abspath(dirname)
        new_filename = os.path.split(real_name)[1]
        hdf_filename = "{basename}-results.h5".format(basename=new_filename)
    else:
        hdf_filename = filename
    if groupname is None:
        groupname = os.path.split(os.path.abspath(dirname))[1]
    # Open actual file
    hdf_file = h5py.File(hdf_filename)
    # Alert the user that we're overwriting this group
    if groupname in hdf_file.keys():
        msg = 'Group "{groupname}" exists. Overwriting.'
        print(msg.format(groupname=groupname))
        del hdf_file[groupname]
        # msg += " Try using the `hdf_groupname` argument"
        # e = exceptions.CreateGroupError(msg.format(groupname=groupname))
        # raise e
    new_group = hdf_file.create_group(groupname)
    # User feedback
    if not prog.quiet:
        print('Saving to HDF5 file {file} in group {group}'.format(
            file=hdf_filename,
            group=groupname)
        )
    return new_group


def import_txm_framesets(*args, **kwargs):
    msg = "This function is ambiguous. Choose from the more specific importers."
    raise NotImplementedError(msg)

def import_ptychography_frameset(directory: str,
                                 hdf_filename=None, hdf_groupname=None):
    """Import a set of images as a new frameset for generating
    ptychography chemical maps based on data collected at ALS beamline
    5.3.2.1

    Arguments
    ---------

    - results_dir : Directory where to look for results. It should
    contain a subdirectory named "tiffs" and a file named
    "energies.txt"

    - hdf_filename : HDF File used to store computed results. If
      omitted, the `directory` basename is used

    - hdf_groupname : String to use for the hdf group of this
    dataset. If omitted or None, the `directory` basename is
    used. Raises an exception if the group exists.
    """
    CURRENT_VERSION = "0.2" # Let's file loaders deal with changes to storage
    # Prepare some filesystem information
    tiff_dir = os.path.join(directory, "tiffs")
    modulus_dir = os.path.join(tiff_dir, "modulus")
    stxm_dir = os.path.join(tiff_dir, "modulus")
    # Prepare the HDF5 file and metadata
    hdf_group = _prepare_hdf_group(filename=hdf_filename,
                                   groupname=hdf_groupname,
                                   dirname=directory)
    hdf_group.attrs["scimap_version"] = CURRENT_VERSION
    hdf_group.attrs["technique"] = "ptychography STXM"
    hdf_group.attrs["beamline"] = "ALS 5.3.2.1"
    hdf_group.attrs["original_directory"] = os.path.abspath(directory)
    # Prepare groups for data
    imported = hdf_group.create_group("imported")
    imported_group = imported.name
    hdf_group["imported"].attrs["level"] = 0
    hdf_group["imported"].attrs["parent"] = ""
    hdf_group["imported"].attrs["default_representation"] = "modulus"
    file_re = re.compile("projection_modulus_(?P<energy>\d+\.\d+)\.tif")
    for filename in os.listdir(modulus_dir):
        # (assumes each image type has the same set of energies)
        # Extract energy from filename
        match = file_re.match(filename)
        if match is None:
            msg = "Could not read energy from filename {}".format(filename)
            raise exceptions.FilenameParseError(msg)
        energy_str = match.groupdict()['energy']
        # All dataset names will be the energy with two decimal places
        energy_set = imported.create_group(energy_key.format(float(energy_str)))
        energy_set.attrs['energy'] = float(energy_str)
        energy_set.attrs['approximate_energy'] = round(float(energy_str), 2)
        energy_set.attrs['pixel_size'] = 4.17
        energy_set.attrs['pixel_unit'] = "nm"
        def add_files(name, template="projection_{name}_{energy}.tif"):
            # Import modulus (total value)
            filename = template.format(name=name, energy=energy_str)
            filepath = os.path.join(tiff_dir, name, filename)
            data = Image.open(filepath)
            energy_set.create_dataset(name, data=data, chunks=True)
        representations = ['modulus', 'phase', 'complex', 'intensity']
        [add_files(name) for name in representations]
        add_files("stxm", template="stxm_{energy}.tif")
    # Create the frameset object
    hdf_filename = hdf_group.file.filename
    hdf_groupname = hdf_group.name
    hdf_group.file.close()
    frameset = XanesFrameset(filename=hdf_filename,
                             groupname=hdf_groupname,
                             edge=None)
    frameset.latest_group = imported_group


def import_fullfield_framesets(directory, hdf_filename=None, flavor='ssrl'):
    """Import all files in the given directory and process into
    framesets. Images are assumed to full-field transmission X-ray
    micrographs."""
    format_classes = {
        '.xrm': XRMFile
    }
    # Prepare list of dataframes to be imported
    file_list = []
    start_time = time()
    for filename in os.listdir(directory):
        # Make sure it's a file
        fullpath = os.path.join(directory, filename)
        if os.path.isfile(fullpath):
            # Import the file if the extension is known
            name, extension = os.path.splitext(filename)
            if extension in format_classes.keys():
                file_list.append(fullpath)
    # Prepare some global data for the import process
    hdf_file = _prepare_hdf_file(filename=hdf_filename, dirname=directory)
    # Find a unique name for the background frames
    formatter = "background_frames_{ctr}"
    counter = 0
    bg_groupname = formatter.format(ctr=counter)
    while bg_groupname in hdf_file.keys():
        counter += 1
        bg_groupname = formatter.format(ctr=counter)
    hdf_file.create_group(bg_groupname)
    if not prog.quiet:
        print('\rCreated background group {}'.format(bg_groupname))
    bg_frameset = XanesFrameset(filename=hdf_filename, groupname=bg_groupname)
    sample_framesets = {}
    # Now do the importing
    total_files = len(file_list)
    while(len(file_list) > 0):
        current_file = file_list[0]
        name, extension = os.path.splitext(current_file)
        # Arrays to keep track of the timestamps in this averaged frame
        starttimes = []
        endtimes = []
        # Average multiple frames together if necessary
        files_to_average = find_average_scans(current_file, file_list, flavor=flavor)
        # Convert to Frame() objects

        def convert_to_frame(file_list):
            frames_to_average = []
            for filepath in file_list:
                Importer = format_classes[extension]
                with Importer(filepath, flavor=flavor) as txm_file:
                    starttimes.append(txm_file.starttime())
                    endtimes.append(txm_file.endtime())
                    frame = TXMFrame(file=txm_file)
                    frames_to_average.append(frame)
            # Average scans
            averaged_frame = average_frames(*frames_to_average)
            return averaged_frame
        averaged_frame = convert_to_frame(files_to_average)
        # Set beginning and end timestamps
        averaged_frame.starttime = min(starttimes)
        averaged_frame.endtime = max(endtimes)
        if averaged_frame.is_background:
            bg_frameset.add_frame(averaged_frame)
        else:
            # Determine which frameset to use or create a new one
            if flavor == 'aps':
                identifier = "{}_{}".format(averaged_frame.sample_name,
                                            averaged_frame.position_name)
            elif flavor == 'ssrl':
                identifier = averaged_frame.sample_name
            if identifier not in sample_framesets.keys():
                # First time seeing this frameset location
                new_frameset = XanesFrameset(filename=hdf_filename,
                                             groupname=identifier)
                new_frameset.active_groupname = 'raw_frames'
                sample_framesets[identifier] = new_frameset
            # Add this frame to the appropriate group
            sample_framesets[identifier].add_frame(averaged_frame)
        for filepath in files_to_average:
            file_list.remove(filepath)
        # Display current progress
        if not prog.quiet:
            status = format_meter(n=total_files - len(file_list),
                                  total=total_files,
                                  elapsed=time() - start_time,
                                  prefix="Importing raw frames: ")
            print("\r", status, end='')
    if not prog.quiet:
        print()  # Blank line to avoid over-writing status message
    # print(' frames: {curr}/{total} [done]'.format(curr=total_files,
    #                                                        total=total_files))
    frameset_list = list(sample_framesets.values())
    # Apply reference correction and convert to absorbance frames
    if not prog.quiet:
        print('Imported samples', [fs.hdf_group().name for fs in frameset_list])
    for frameset in frameset_list:
        frameset.apply_references(bg_groupname=bg_groupname)
    # Apply magnification (zoom) correction
    if flavor in ['ssrl']:
        for frameset in frameset_list:
            frameset.correct_magnification()
    else:
        print('Skipped magnification correction')
    # Remove dead or hot pixels
    if flavor in ['aps']:
        sigma = 9
        for fs in frameset_list:
            for frame in prog(fs, 'Removing pixels beyond {}σ'.format(sigma)):
                frame.remove_outliers(sigma=sigma)
    return frameset_list


def find_average_scans(filename, file_list, flavor='ssrl'):
    """Scan the filenames in `file_list` and see if there are multiple
    subframes per frame."""
    basename = os.path.basename(filename)
    dirname = os.path.dirname(filename)
    if flavor == 'ssrl':
        avg_regex = re.compile(r"(\d+)of(\d+)")
        # Since this regex will be passed to sub later,
        # we must use it as a template first
        serial_template = "_{fmt}{length}_ref_"
        serial_string = serial_template.format(fmt="\d", length="{6}")
        assert serial_string == "_\d{6}_ref_"
        serial_regex = re.compile(serial_string)
        # Look for average scans
        re_result = avg_regex.search(basename)
        if re_result:
            # Use regular expressions to determine the other files
            total = int(re_result.group(2))
            current_files = []
            for current in range(1, total + 1):
                new_regex = r"0*{current}of0*{total}".format(
                    current=current, total=total)
                filename_restring = avg_regex.sub(new_regex, basename)
                # Replace serial number if necessary (reference frames only)
                filename_restring = serial_regex.sub(serial_template,
                                                     filename_restring)
                filename_restring = filename_restring.format(fmt="\d",
                                                             length="{6}")
                # Find the matching filenames in the list
                filepath_regex = re.compile(os.path.join(dirname, filename_restring))
                for filepath in file_list:
                    if filepath_regex.match(filepath):
                        current_files.append(filepath)
                        break
            if not len(current_files) == total:
                msg = "Could not find all all files to average, only found {}"
                raise exceptions.FrameFileNotFound(msg.format(current_files))
        else:
            current_files = [filename]
    elif flavor == "aps":
        current_files = [filename]
    return current_files
