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

from .xradia import XRMFile
from .xanes_frameset import XanesFrameset
from .frame import TXMFrame, average_frames
from utilities import prog
import exceptions


def import_txm_framesets(directory, hdf_filename=None, flavor='ssrl'):
    """Import all files in the given directory and process into framesets."""
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
    if hdf_filename is None:
        real_name = os.path.abspath(directory)
        new_filename = os.path.split(real_name)[1]
        hdf_filename = os.path.join(
            directory,
            "{basename}.hdf".format(basename=new_filename)
        )
    if os.path.exists(hdf_filename):
        msg = "File {} already exists. Try using the `hdf_filename` argument"
        raise exceptions.FileExistsError(
            msg.format(os.path.basename(hdf_filename))
        )
    if not prog.quiet:
        print('Saving to HDF5 file: {}'.format(hdf_filename))
    hdf_file = h5py.File(hdf_filename)
    # Find a unique name for the background frames
    formatter = "background_frames_{ctr}"
    counter = 0
    bg_groupname = formatter.format(ctr=counter)
    while bg_groupname in hdf_file.keys():
        counter += 1
        bg_groupname = formatter.format(ctr=counter)
    hdf_file.create_group(bg_groupname)
    if not prog.quiet:
        print('Created background group {}'.format(bg_groupname))
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
            print(status, end='\r')
    if not prog.quiet:
        print()  # Blank line to avoid over-writing status message
    # print(' frames: {curr}/{total} [done]'.format(curr=total_files,
    #                                                        total=total_files))
    frameset_list = list(sample_framesets.values())
    # Apply reference collection and convert to absorbance frames
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
    # Identify particles
    for frameset in frameset_list:
        frameset.label_particles()
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
