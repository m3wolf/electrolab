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
# along with scimap.  If not, see <http://www.gnu.org/licenses/>.

import os
import re
import chunk
import codecs
from collections import namedtuple
import zipfile
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from . import exceptions, default_units as units
from .tube import tubes
from .utilities import twotheta_to_q, q_to_twotheta
from .utilities import shape, Pixel


def adapter_from_filename(filename, *args, **kwargs):
    """Use the given filename and extension to retrieve to appropriate
    file adapter. Additional args and kwargs are passed to the
    constructor for the adapter, which allows for data to be passed
    that may not be included in the data file itself (eg radiation
    wavelength).
    """
    FILE_ADAPTERS = {
        '.xye': BrukerXyeFile,
        '.plt': BrukerPltFile,
        '.dat': FullProfDataFile,
        '.brml': BrukerBrmlFile,
        '.gfrm': BrukerGfrmFile,
    }
    # Determine file type from extension
    filebase, extension = os.path.splitext(filename)
    # Prepare adapter for importing the file
    try:
        Adapter = FILE_ADAPTERS[extension]
    except KeyError:
        # Unknown file format, raise exception
        msg = 'Unknown file format {extension} ({filename}).'
        msg = msg.format(extension=extension, filename=filename)
        raise exceptions.FileFormatError(msg)
    else:
        adapter = Adapter(filename, *args, **kwargs)
        return adapter


class XRDAdapter():
    """Allows for consistent importing of XRD data from different sources."""
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # For compatibility as a context manager
        pass

    @property
    def sample_name(self):
        raise NotImplementedError(self.__class__)

    def scattering_lengths(self):
        raise NotImplementedError(self.__class__)

    def intensities(self):
        raise NotImplementedError(self.__class__)

    @property
    def detector_image(self):
        raise NotImplementedError(self.__class__)


class BrukerXyeFile(XRDAdapter):
    """File adapter for Bruker's .xye format.

    Arguments
    ---------
    - wavelength : X-ray wavelength, in angstroms if no units are
      attached to it. If omitted, the file will be parsed for anode type
      and looked-up in X-ray tube file.
    """
    _wavelength = None
    def __init__(self, filename, wavelength=None):
        self._wavelength = wavelength
        self.filename = filename

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # For compatibility with more complex file types
        pass

    @property
    def sample_name(self):
        return self.filename

    def scattering_lengths(self, wavelength=None):
        if wavelength is None:
            wavelength = self.wavelength
        twotheta = np.array(self._dataframe.index)
        q = twotheta_to_q(twotheta, wavelength=wavelength)
        return q

    def intensities(self):
        df = self._dataframe
        return df['counts'].values

    @property
    def wavelength(self):
        # self._wavelength is set if it was passed to constructor
        if self._wavelength:
            wl = self._wavelength
        else:
            # Try and find anode type from file
            with open(self.filename, mode='r') as f:
                header = f.readline()
            match = re.search(r"Anode: ([A-Za-z]+)", header)
            if match:
                anode = match.group(1)
                try:
                    wl = tubes[anode].kalpha
                except KeyError:
                    msg = 'No definition for {} X-ray tube.'.format(anode)
                    raise exceptions.FileFormatError(msg)
            else:
                msg = 'Wavelength not found in file, pass `wavelength`to constructor.'                  
                raise exceptions.DataFormatError(msg)
                
        return wl

    @property
    def _dataframe(self):
        df = pd.read_csv(self.filename,
                         names=['2theta', 'counts', 'error'],
                         sep='\s+', index_col=0, comment="'")
        return df


class BrukerPltFile(XRDAdapter):
    def __init__(self, filename):
        self.filename = filename
    
    @property
    def sample_name(self):
        return self.filename
    
    def _dataframe(self):
        df = pd.read_csv(self.filename,
                         names=['2theta', 'counts'],
                         sep=' ', index_col=0, comment="!")
        return df
    
    def intensities(self):
        data = self._dataframe()
        return data['counts'].values
    
    def scattering_lengths(self, wavelength):
        """Return scattering length (q) for all Datum elements in the file."""
        # Find all Datum entries in data tree
        df = self._dataframe()
        two_theta = df.index
        q = twotheta_to_q(two_theta=two_theta, wavelength=wavelength)
        return q


class BrukerBrmlFile(XRDAdapter):
    def __init__(self, filename):
        self._zf = zipfile.ZipFile(filename)
        data_file = self._zf.open('Experiment0/RawData0.xml')
        self._dataTree = ElementTree.parse(data_file)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # Close files on disk
        self._zf.close()

    @property
    def sample_name(self):
        nameElement = self._dataTree.find('.//InfoItem[@Name="SampleName"]')
        name = nameElement.get('Value')
        return name

    def scattering_lengths(self, wavelength=None):
        """Return scattering length (q) for all Datum elements in the file.
        
        The wavelength is retrieved from the file; the argument is
        included only for interface compatibility.

        Returns
        -------
        q : np.ndarray
          The scattering lengths in reciprocal angstroms, derived from
          measured 2θ values

        """
        # Find all Datum entries in data tree
        data = self._dataTree.findall('.//Datum')
        two_theta = np.array([float(d.text.split(',')[2]) for d in data])
        q = twotheta_to_q(two_theta=two_theta, wavelength=self.wavelength)
        return q

    def intensities(self):
        """Return photon counts for all elements in the file."""
        data = self._dataTree.findall('.//Datum')
        counts = np.array([int(d.text.split(',')[4]) for d in data])
        return counts

    @property
    def wavelength(self):
        """Wavelength of the incoming radiation in Angstroms."""
        tubeElement = self._dataTree.find('.//WaveLengthAverage')
        unit_str = tubeElement.get('Unit')
        num = float(tubeElement.get('Value'))
        if unit_str == 'Å': # Special character copied from brml file
            wavelength = num * units.angstrom
        else:
            wavelength = num * getattr(units, unit_str)
        result = float(wavelength / units.angstrom)
        return result



ByteField = namedtuple('ByteField', ('name', 'type', 'length'))

class BrukerRawFile(XRDAdapter):
    ENCODING = 'latin-1'
    HEADER_BYTES = [
        ByteField('raw_id', 'str', 10),
        ByteField('meas_flag', 'int', 1),
        ByteField('no_of_tot_meas_ranges', 'int', 1),
        ByteField('date', 'str', 12),
        ByteField('time', 'str', 8),
        ByteField('junk', 'str', 29),
        ByteField('user', 'str', None),
        ByteField('site', 'str', None),
        ByteField('sample_name', 'str', None),
    ]

    def __init__(self, filename):
        # Load file and save data
        with open(filename, 'rb') as f:
            self.data = f.read()
        formatString = self.data[:3].decode()
        if formatString != 'RAW':
            msg = "Not in Bruker RAW4.00 format"
            raise exceptions.FileFormatError(msg)
        self.parse_header()

    @property
    def sample_name(self):
        self.header['sample_name']

    def byte_fields(self, field_list, offset=0):
        """
        Generator steps through self.data starting at offset and returns
        data. Each item in field list should have a .type and .length
        attribute.
        """
        currentPos = offset
        FieldValue = namedtuple('FieldValue', ('name', 'value'))
        for field in field_list:
            # Extract field value from data
            if field.type == 'int':
                start = currentPos
                end = currentPos + field.length
                value = self.data[start:end]
                currentPos += field.length
            elif field.type == 'str' and field.length is None:
                # Variable length string
                value, length = self.extract_string(currentPos)
                currentPos += length
            elif field.type == 'str':
                # Fixed length string
                start = currentPos
                end = currentPos + field.length
                value = codecs.decode(
                    self.data[start:end],
                    self.ENCODING
                )
                currentPos += field.length
            result = FieldValue(name=field.name, value=value)
            yield(result)

    def parse_header(self):
        self.header = {}
        for field in self.byte_fields(self.HEADER_BYTES):
            self.header[field.name] = field.value

    def extract_string(self, offset):
        marker = offset
        # Chop bytestring up by looking for a newline
        while chr(self.data[marker]) != '\n':
            marker += 1
        byteString = self.data[offset:marker]
        newString = codecs.decode(byteString, self.ENCODING)
        return newString, marker - offset

##############################################
# GADDS 2D Laboratory diffractometer by Bruker
##############################################
byte_value = namedtuple('byte_value', ('name', 'length'))
default_length = 80
gfrm_header_definitions = [
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

class BrukerGfrmFile():
    def __init__(self, filename):
        self.filename = filename

    def detector_image(self, mask="auto"):
        """Return a two-dimensional image as captured by an area detector."""
        return self.import_gadds_frame()

    def _read_header(self, header_bytes):
        """The GFRM header is a byte string with a series of meta data of the
        form "KEY: VALUE" with lots of whitespace padding. The dictionary
        `header_definitions` hold the descriptions of how long each pair
        is.
        """
        data = {}
        cursor = 0
        for definition in gfrm_header_definitions:
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

    def import_gadds_frame(self, mask_radius="auto"):
        """Import a two-dimensional X-ray diffraction pattern from a
        GADDS system detector, typically using the .gfrm
        extension.
        """
        filename = self.filename
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
            metadata = self._read_header(header)
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


# class FullProfDataFile():
#     def __init__(self, filename):
#         self.filename = filename

#     def write_diffractogram(self, two_theta, intensities):
#         df = 
        
#     def write_dataframe(self, dataframe):
#         """
#         Write the 2-theta, counts data for the given scan in a format
#         suitable for feeding into the FullProf refinement program.
#         """
#         result = dataframe.to_csv(self.filename,
#                                   columns=['counts'],
#                                   sep=' ',
#                                   header=False)
#         return result
