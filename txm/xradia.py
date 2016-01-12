from collections import namedtuple
import struct
import os
import re

from PIL import OleFileIO
import numpy as np

import exceptions

def decode_ssrl_params(filename):
    """Accept the filename of an XRM file and return sample parameters as
    a dictionary."""
    # Beamline 6-2c at SSRL
    ssrl_regex_bg = re.compile(
        'rep(\d{2})_(\d{6})_ref_[0-9]+_([a-zA-Z0-9_]+)_([0-9.]+)_eV_(\d{3})of(\d{3})\.xrm'
    )
    ssrl_regex_sample = re.compile(
        'rep(\d{2})_[0-9]+_([a-zA-Z0-9_]+)_([0-9.]+)_eV_(\d{3})of(\d{3}).xrm'
    )
    # Check for background frames
    bg_result = ssrl_regex_bg.search(filename)
    if bg_result:
        params = {
            'date_string': '',
            'sample_name': bg_result.group(3).strip("_"),
            'position_name': '',
            'is_background': True,
            'energy': float(bg_result.group(4)),
        }
    else:
        sample_result = ssrl_regex_sample.search(filename)
        params = {
            'date_string': '',
            'sample_name': sample_result.group(2).strip("_"),
            'position_name': '',
            'is_background': False,
            'energy': float(sample_result.group(3)),
        }
    return params

# Some of the byte decoding was taken from
# https://github.com/data-exchange/data-exchange/blob/master/xtomo/src/xtomo_reader.py

class XRMFile():
    """Single X-ray micrscopy frame created using XRadia XRM format."""
    aps_regex = re.compile("(\d{8})_([a-zA-Z0-9_]+)_([a-zA-Z0-9]+)_(\d{4}).xrm")
    def __init__(self, filename, flavor):
        self.filename = filename
        self.flavor = flavor
        self.ole_file = OleFileIO.OleFileIO(self.filename)
        # Filename parameters
        params = self.parameters_from_filename()
        self.sample_name = params['sample_name']
        self.position_name = params['position_name']
        self.is_background = params['is_background']

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.ole_file.close()

    def __str__(self):
        return os.path.basename(self.filename)

    def __repr__(self):
        return "<XRMFile: '{}'>".format(os.path.basename(self.filename))

    def parameters_from_filename(self):
        """Determine various metadata from the frames filename (sample name etc)."""
        if self.flavor == 'aps':
            # APS beamline 8-BM-B
            result = self.aps_regex.search(self.filename)
            params = {
                'date_string': result.group(1),
                'sample_name': result.group(2),
                'position_name': result.group(3),
                'is_background': result.group(3) == 'bkg',
                'energy': float(result.group(4)),
            }
        elif self.flavor == 'ssrl':
            params = decode_ssrl_params(self.filename)
        else:
            msg = "Unknown flavor for filename: {}"
            raise exceptions.FileFormatError(msg.format(self.filename))
        return params

    def ole_value(self, stream, fmt=None):
        """Get arbitrary data from the ole file and convert from bytes."""
        stream_bytes = self.ole_file.openstream(stream).read()
        if fmt is not None:
            stream_value = struct.unpack(fmt, stream_bytes)[0]
        else:
            stream_value = stream_bytes
        return stream_value

    def energy(self):
        """Beam energy in electronvoltes."""
        # Try reading from file first
        energy = self.ole_value('ImageInfo/Energy', '<f')
        if not energy > 0:
            # if not, read from filename
            re_result = re.search("(\d+\.?\d?)_?eV", self.filename)
            if re_result:
                energy = float(re_result.group(1))
            else:
                msg = "Could not read energy for file {}"
                raise exceptions.FileFormatError(msg.format(self.filename))
        return energy

    def image_data(self):
        """TXM Image frame."""
        # Figure out byte size
        dimensions = self.image_size()
        num_bytes = dimensions.horizontal * dimensions.vertical
        # Determine format string
        image_dtype = self.image_dtype()
        if image_dtype == 'uint16':
            fmt_str = "<{}h".format(num_bytes)
        elif image_dtype == 'float32':
            fmt_str = "<{}f".format(num_bytes)
        # Return decoded image data
        stream = self.ole_file.openstream('ImageData1/Image1')
        img_data = struct.unpack(fmt_str, stream.read())
        img_data = np.reshape(img_data, dimensions)
        return img_data

    # def is_background(self):
    #     """Look at the file name for clues to whether this is a background
    #     frame."""
    #     result = re.search('bkg|_ref_', self.filename)
    #     return bool(result)

    def sample_position(self):
        position = namedtuple('position', ('x', 'y', 'z'))
        x = self.ole_value('ImageInfo/XPosition', '<f')
        y = self.ole_value('ImageInfo/YPosition', '<f')
        z = self.ole_value('ImageInfo/ZPosition', '<f')
        return position(x, y, z)

    def binning(self):
        vertical = self.ole_value('ImageInfo/VerticalalBin', '<L')
        horizontal = self.ole_value('ImageInfo/HorizontalBin', '<L')
        binning = namedtuple('binning', ('horizontal', 'vertical'))
        return binning(horizontal, vertical)

    def image_dtype(self):
        dtypes = {
            5: 'uint16',
            10: 'float32',
        }
        dtype_number = self.ole_value('ImageInfo/DataType', '<1I')
        return dtypes[dtype_number]

    def image_size(self):
        resolution = namedtuple('dimensions', ('horizontal', 'vertical'))
        horizontal = self.ole_value('ImageInfo/ImageWidth', '<I')
        vertical = self.ole_value('ImageInfo/ImageHeight', '<I')
        return resolution(horizontal=horizontal, vertical=vertical)
