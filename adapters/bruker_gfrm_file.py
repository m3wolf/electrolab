"""Collection of utilities for interacting with Bruker instruments and data."""
import chunk
from collections import namedtuple

import numpy as np

from utilities import shape, Pixel

byte_value = namedtuple('byte_value', ('name', 'length'))


class BrukerGfrmFile():
    sample_name = "TODO: sample name"

    def __init__(self, filename):
        self.filename = filename

    def image(self, mask="auto"):
        return import_gadds_frame(self.filename)

    @property
    def dataframe(self):
        print('Diffractogram coming soon')


default_length = 80
header_definitions = [
    byte_value('???', 72),
    byte_value('version', default_length),
    byte_value('header_blocks', default_length),
    byte_value('type', default_length),
    byte_value('site', default_length),
    byte_value('model', default_length),
    byte_value('user', default_length),
    byte_value('sample', default_length),
    byte_value('setname', default_length),
    byte_value('run', default_length),
    byte_value('sample_number', default_length),
    byte_value('title1', default_length),
    byte_value('title2', default_length),
    byte_value('title3', default_length),
    byte_value('title4', default_length),
    byte_value('title5', default_length),
    byte_value('title6', default_length),
    byte_value('title7', default_length),
    byte_value('title8', default_length),
    byte_value('ncounts', default_length),
    byte_value('noverfl', default_length),
    byte_value('minimum', default_length),
    byte_value('maximum', default_length),
    byte_value('nontime', default_length),
    byte_value('nlate', default_length),
    byte_value('filename', default_length),
    byte_value('created', default_length),
    byte_value('cumulat', default_length),
    byte_value('elapsdr', default_length),
    byte_value('elapsda', default_length),
    byte_value('oscilla', default_length),
    byte_value('nsteps', default_length),
    byte_value('range', default_length),
    byte_value('start', default_length),
    byte_value('increment', default_length),
    byte_value('number', default_length),
    byte_value('nframes', default_length),
    byte_value('angles', default_length),
    byte_value('nover64', default_length),
    byte_value('npixelb', default_length),
    byte_value('nrows', default_length),
    byte_value('ncols', default_length),
    byte_value('word_order', default_length),
    byte_value('long_oder', default_length),
    byte_value('target', default_length),
    byte_value('sourcek', default_length),
    byte_value('sourcem', default_length),
    byte_value('filter', default_length),
    byte_value('cell1', default_length),
    byte_value('cell2', default_length),
    byte_value('matrix1', default_length),
    byte_value('matrix2', default_length),
    byte_value('lowtemp', default_length),
    byte_value('zoom', default_length),
    byte_value('center', default_length),
    byte_value('distance', default_length),
    byte_value('trailer', default_length),
    byte_value('compres', default_length),
    byte_value('linear', default_length),
    byte_value('phd', default_length),
    byte_value('preamp', default_length),
    byte_value('floodfile', default_length),
    byte_value('warpfile', default_length),
    byte_value('wavelengths', default_length),
    byte_value('maxxy', default_length),
    byte_value('axis', default_length),
    byte_value('ending', default_length),
    byte_value('detpar1', default_length),
    byte_value('detpar1', default_length),
    byte_value('lut', default_length),
    byte_value('display_limit', default_length),
    byte_value('program', default_length),
    byte_value('rotation', default_length),
    byte_value('bitmask', default_length),
    byte_value('octmask1', default_length),
    byte_value('octmask2', default_length),
    byte_value('esdcell1', default_length),
    byte_value('esdcell2', default_length),
    byte_value('detector_type', default_length),
    byte_value('nexp', default_length),
    byte_value('ccdparm', default_length),
    byte_value('chem', default_length),
    byte_value('morph', default_length),
    byte_value('ccolor', default_length),
    byte_value('csize', default_length),
    byte_value('dnsmet', default_length),
    byte_value('dark', default_length),
    byte_value('autorng', default_length),
    byte_value('zeroadj', default_length),
    byte_value('xtrans', default_length),
    byte_value('hkl&xy', default_length),
    byte_value('axes2', default_length),
    byte_value('ending2', default_length),
]


def read_header(header_bytes):
    """The GFRM header is a byte string with a series of meta data of the
    form "KEY: VALUE" with lots of whitespace padding. The dictionary
    `header_definitions` hold the descriptions of how long each pair
    is.
    """
    data = {}
    cursor = 0
    for definition in header_definitions:
        # Figure out where to stop
        end = cursor + definition.length
        # Get the data
        s = header_bytes[cursor:end]
        cursor = end
        # key_value = s.split(b':')
        key = s[0:7]
        sep = s[7]
        value = s[8:]
        if sep != 58:
            # Entry does not contain actual data if separator is not 58 (b":")
            continue
        # Clean up whitespace and convert to unicode strings
        key = key.decode().strip()
        value = value.decode().strip()
        # Add it to the dictionary
        data[definition.name] = value
    return data


def import_gadds_frame(filename, mask_radius="auto"):
    """Import a two-dimensional X-ray diffraction pattern from a
    GADDS system detector, typically using the .gfrm
    extension.
    """
    with open(filename, 'rb') as f:
        raw_chunk = chunk.Chunk(f)
        # Find out how big the header is
        raw_chunk.seek(152)
        preamble = raw_chunk.read(50)
        assert preamble[0:8] == b"HDRBLKS:"
        hdrblks = int(preamble[9:])
        raw_chunk.seek(0)
        # Read in the header
        header = raw_chunk.read(hdrblks * 512 - 8)
        # Determine image dimensions from header
        metadata = read_header(header)
        frame_shape = shape(rows=int(metadata['nrows']),
                            columns=int(metadata['ncols']))
        data_length = frame_shape.rows * frame_shape.columns
        data_bytes = raw_chunk.read(data_length)
        # leftover = raw_chunk.read() # Not used
    # Prepare an array of the image data
    data = np.fromstring(data_bytes, dtype=np.dtype('u1'))
    data = data.reshape(frame_shape)
    # Apply a round mask
    x, y = np.ogrid[:frame_shape.rows, :frame_shape.columns]
    c = Pixel(vertical=frame_shape.rows / 2,
              horizontal=frame_shape.columns / 2)

    # convert cartesian --> polar coordinates
    dx = x - c.horizontal
    dy = y - c.vertical
    r2 = (dx * dx) + (dy * dy)

    # Determine radius from image dimensions
    if mask_radius == "auto":
        radius = 0.95 * min(frame_shape) / 2
    circmask = np.logical_not(r2 <= radius * radius)
    masked_data = np.ma.array(data, mask=circmask)
    return masked_data
