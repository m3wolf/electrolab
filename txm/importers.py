import os
import h5py
import re

from .xradia import XRMFile
from .xanes_frameset import XanesFrameset
from .frame import TXMFrame, average_frames
import exceptions

def import_txm_framesets(directory, hdf_filename=None, flavor='ssrl'):
    """Import all files in the given directory and process into framesets."""
    format_classes = {
        '.xrm': XRMFile
    }
    # Prepare list of dataframes to be imported
    file_list = []
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
    print('Created background group {}'.format(bg_groupname))
    bg_frameset = XanesFrameset(filename=hdf_filename, groupname=bg_groupname)
    sample_framesets = {}
    # Now do the importing
    total_files = len(file_list)
    while(len(file_list) > 0):
        current_file = file_list[0]
        name, extension = os.path.splitext(current_file)
        # Average multiple frames together if necessary
        files_to_average = find_average_scans(current_file, file_list, flavor=flavor)
        frames_to_average = []
        # Convert to Frame() objects
        for filepath in files_to_average:
            Importer = format_classes[extension]
            with Importer(filepath, flavor=flavor) as txm_file:
                frame = TXMFrame(file=txm_file)
                frames_to_average.append(frame)
        # Average scans
        averaged_frame = average_frames(*frames_to_average)
        # Remove from queue and add to frameset
        for filepath in files_to_average:
            file_list.remove(filepath)
        if averaged_frame.is_background:
            bg_frameset.add_frame(averaged_frame)
        else:
            # Determine which frameset to use or create a new one
            identifier = "{}_{}".format(averaged_frame.sample_name,
                                        averaged_frame.position_name)
            if not identifier in sample_framesets.keys():
                # First time seeing this frameset location
                new_frameset = XanesFrameset(filename=hdf_filename,
                                             groupname=identifier)
                new_frameset.active_groupname = 'raw_frames'
                sample_framesets[identifier] = new_frameset
            # Add this frame to the appropriate group
            sample_framesets[identifier].add_frame(averaged_frame)
        # Display current progress
        template = 'Averaging frames: {curr}/{total} ({percent:.0f}%)'
        status = template.format(
            curr=total_files - len(file_list),
            total=total_files,
            percent=(1 - (len(file_list)/total_files))*100
        )
        print(status, end='\r')
    print('Importing raw frames: {curr}/{total} [done]'.format(curr=total_files,
                                                           total=total_files))
    frameset_list = list(sample_framesets.values())
    # Apply reference collection and convert to absorbance frames
    print('Imported samples', [fs.hdf_group().name for fs in frameset_list])
    for frameset in frameset_list:
        frameset.subtract_background(bg_groupname)
    return frameset_list


def find_average_scans(filename, file_list, flavor='ssrl'):
    """Scan the filenames in `file_list` and see if there are multiple
    subframes per frame."""
    basename = os.path.basename(filename)
    dirname = os.path.dirname(filename)
    if flavor == 'ssrl':
        avg_regex = re.compile("(\d+)of(\d+)")
        serial_string = "_\d{6}_ref_"
        serial_regex = re.compile(serial_string)
        # Look for average scans
        re_result = avg_regex.search(basename)
        if re_result:
            # Use regular expressions to determine the other files
            total = int(re_result.group(2))
            current_files = []
            for current in range(1, total+1):
                new_regex = "0*{current}of0*{total}".format(
                    current=current, total=total)
                filename_restring = avg_regex.sub(new_regex, basename)
                # Replace serial number if necessary (reference frames only)
                filename_restring = serial_regex.sub(serial_string, filename_restring)
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
