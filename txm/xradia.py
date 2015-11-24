from collections import namedtuple
import struct
import os
import re

from PIL import OleFileIO
import numpy as np

import exceptions

# Some of the byte decoding was taken from
# https://github.com/data-exchange/data-exchange/blob/master/xtomo/src/xtomo_reader.py

class XRMFile():
    """Single X-ray micrscopy frame."""
    def __init__(self, filename):
        self.filename = filename
        self.ole_file = OleFileIO.OleFileIO(self.filename)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.ole_file.close()

    def __str__(self):
        return os.path.basename(self.filename)

    def __repr__(self):
        return "<XRMFile: '{}'>".format(os.path.basename(self.filename))

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

    def is_background(self):
        """Look at the file name for clues to whether this is a background
        frame."""
        result = re.search('bkg|_ref_', self.filename)
        return bool(result)

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
