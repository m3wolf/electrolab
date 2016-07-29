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

import sys
import re
from os import SEEK_SET
import time
from datetime import date, datetime
from collections import OrderedDict

import units
import pandas as pd
import numpy as np

"""Code to read in data files from Bio-Logic instruments. Class for
reading MPR files taken from
https://github.com/chatcannon/galvani/blob/master/galvani/BioLogic.py"""

if sys.version_info.major <= 2:
    str3 = str
    from string import maketrans
else:
    def str3(b):
        str(b, encoding='ascii')
    maketrans = bytes.maketrans


def fieldname_to_dtype(fieldname):
    """Converts a column header from the MPT file into a tuple of
    canonical name and appropriate numpy dtype"""

    if fieldname == 'mode':
        return ('mode', np.uint8)
    elif fieldname in ("ox/red", "error", "control changes", "Ns changes",
                       "counter inc."):
        return (fieldname, np.bool_)
    elif fieldname in ("time/s", "P/W", "(Q-Qo)/mA.h", "x", "control/V",
                       "control/V/mA", "(Q-Qo)/C", "dQ/C", "freq/Hz",
                       "|Ewe|/V", "|I|/A", "Phase(Z)/deg", "|Z|/Ohm",
                       "Re(Z)/Ohm", "-Im(Z)/Ohm", "d(Q-Qo)/dE/mA.h/V"):
        return (fieldname, np.float_)
    # N.B. I'm not sure what 'Ns' is as in the only file I have with that
    # header it never has any value other than '0'
    elif fieldname in ("cycle number", "I Range", "Ns"):
        return (fieldname, np.int_)
    elif fieldname in ("dq/mA.h", "dQ/mA.h"):
        return ("dQ/mA.h", np.float_)
    elif fieldname in ("I/mA", "<I>/mA"):
        return ("I/mA", np.float_)
    elif fieldname in ("Ewe/V", "<Ewe>/V"):
        return ("Ewe/V", np.float_)
    else:
        raise ValueError("Invalid column header: %s" % fieldname)


def comma_converter(float_string):
    """Convert numbers to floats whether the decimal point is '.' or ','"""
    # Check for 'XXX' as bad data
    if float_string == b'XXX':
        return np.nan
    else:
        trans_table = maketrans(b',', b'.')
        return float(float_string.translate(trans_table))


def process_mpt_headers(headers):
    """Process a list of text lines containing metadata from an MPT
    file. Look for certain patterns and return them as a dictionary."""
    mass_re = re.compile('^Mass of active material : ([0-9.]+) ([kmµ]?g)')
    metadata = {}
    for line in headers:
        # Check for active mass
        match = mass_re.match(line)
        if match:
            mass_num, mass_unit = match.groups()
            # We found the match, now save it
            metadata['mass'] = units.unit(mass_unit)(float(mass_num))
        # Check for starttime
        if line[0:25] == "Acquisition started on : ":
            date_string = line[25:].strip()
            start = datetime.strptime(date_string, "%m/%d/%Y %H:%M:%S")
            metadata['start_time'] = start
    return metadata


class MPTFile():
    """Simple function to open MPT files as csv.DictReader objects

    Checks for the correct headings, skips any comments and returns a
    csv.DictReader object and a list of comments
    """
    encoding = "latin-1"

    def __init__(self, filename):
        self.filename = filename

        with open(self.filename, encoding=self.encoding) as mpt_file:
            magic = next(mpt_file)
            valid_magics = [
                'EC-Lab ASCII FILE',
                'BT-Lab ASCII FILE',
            ]
            if magic.rstrip() not in valid_magics:
                msg = "Bad first line for EC-Lab file: '{}'"
                raise ValueError(msg.format(magic))

            nb_headers_match = re.match('Nb header lines : (\d+)\s*$',
                                        next(mpt_file))
            nb_headers = int(nb_headers_match.group(1))
            if nb_headers < 3:
                raise ValueError("Too few header lines: %d" % nb_headers)

            # The 'magic number' line, the 'Nb headers' line and the
            # column headers make three lines. Every additional line
            # is a comment line.
            header = [next(mpt_file) for i in range(nb_headers - 3)]

        self.metadata = process_mpt_headers(header)

        self.load_csv(self.filename)
        # self.data = csv.DictReader(self.mpt_file, dialect='excel-tab')

    def load_csv(self, filename, *args, **kwargs):
        """Wrapper around pandas read_csv that filters out crappy data"""
        # Determine start of data
        with open(filename, encoding=self.encoding) as dataFile:
            # The second line states how long the header is
            header_match = re.match("Nb header lines : (\d+)",
                                    dataFile.readlines()[1])
            headerLength = int(header_match.groups()[0]) - 1
            # headerLength = int(dataFile.readlines()[1][18:20]) - 1
        # Skip all the initial metadata
        df = pd.read_csv(filename,
                         *args,
                         skiprows=headerLength,
                         na_values='XXX',
                         encoding=self.encoding,
                         sep='\t',
                         error_bad_lines=False,
                         **kwargs)
        self.dataframe = df
        return df

    def active_mass(self):
        """Read the mpt file and extract the sample mass"""
        return self.metadata['mass']


VMPmodule_hdr = np.dtype([('shortname', 'S10'),
                          ('longname', 'S25'),
                          ('length', '<u4'),
                          ('version', '<u4'),
                          ('date', 'S8')])


def VMPdata_dtype_from_colIDs(colIDs):
    dtype_dict = OrderedDict()
    flags_dict = OrderedDict()
    flags2_dict = OrderedDict()
    for colID in colIDs:
        if colID in (1, 2, 3, 21, 31, 65):
            dtype_dict['flags'] = 'u1'
            if colID == 1:
                flags_dict['mode'] = (np.uint8(0x03), np.uint8)
            elif colID == 2:
                flags_dict['ox/red'] = (np.uint8(0x04), np.bool_)
            elif colID == 3:
                flags_dict['error'] = (np.uint8(0x08), np.bool_)
            elif colID == 21:
                flags_dict['control changes'] = (np.uint8(0x10), np.bool_)
            elif colID == 31:
                flags_dict['Ns changes'] = (np.uint8(0x20), np.bool_)
            elif colID == 65:
                flags_dict['counter inc.'] = (np.uint8(0x80), np.bool_)
            else:
                raise NotImplementedError("flag %d not implemented" % colID)
        elif colID in (131,):
            dtype_dict['flags2'] = '<u2'
            if colID == 131:
                flags2_dict['??'] = (np.uint16(0x0001), np.bool_)
        elif colID == 4:
            dtype_dict['time/s'] = '<f8'
        elif colID == 5:
            dtype_dict['control/V/mA'] = '<f4'
        # 6 is Ewe, 77 is <Ewe>, I don't see the difference
        elif colID in (6, 77):
            dtype_dict['Ewe/V'] = '<f4'
        # Can't see any difference between 7 and 23
        elif colID in (7, 23):
            dtype_dict['dQ/mA.h'] = '<f8'
        # 76 is <I>, 8 is either I or <I> ??
        elif colID in (8, 76):
            dtype_dict['I/mA'] = '<f4'
        elif colID == 11:
            dtype_dict['I/mA'] = '<f8'
        elif colID == 19:
            dtype_dict['control/V'] = '<f4'
        elif colID == 24:
            dtype_dict['cycle number'] = '<f8'
        elif colID == 32:
            dtype_dict['freq/Hz'] = '<f4'
        elif colID == 33:
            dtype_dict['|Ewe|/V'] = '<f4'
        elif colID == 34:
            dtype_dict['|I|/A'] = '<f4'
        elif colID == 35:
            dtype_dict['Phase(Z)/deg'] = '<f4'
        elif colID == 36:
            dtype_dict['|Z|/Ohm'] = '<f4'
        elif colID == 37:
            dtype_dict['Re(Z)/Ohm'] = '<f4'
        elif colID == 38:
            dtype_dict['-Im(Z)/Ohm'] = '<f4'
        elif colID == 39:
            dtype_dict['I Range'] = '<u2'
        elif colID == 70:
            dtype_dict['P/W'] = '<f4'
        elif colID == 434:
            dtype_dict['(Q-Qo)/C'] = '<f4'
        elif colID == 435:
            dtype_dict['dQ/C'] = '<f4'
        else:
            raise NotImplementedError("column type %d not implemented" % colID)
    return np.dtype(list(dtype_dict.items())), flags_dict, flags2_dict


def read_VMP_modules(fileobj, read_module_data=True):
    """Reads in module headers in the VMPmodule_hdr format. Yields a dict
    with the headers and offset for each module.

    N.B. the offset yielded is the offset to the start of the data
    i.e. after the end of the header. The data runs from (offset) to
    (offset+length)
    """
    while True:
        module_magic = fileobj.read(len(b'MODULE'))
        if len(module_magic) == 0:  # end of file
            raise StopIteration
        elif module_magic != b'MODULE':
            msg = "Found {}, expecting start of new VMP MODULE"
            raise ValueError(msg.format(module_magic))

        hdr_bytes = fileobj.read(VMPmodule_hdr.itemsize)
        if len(hdr_bytes) < VMPmodule_hdr.itemsize:
            raise IOError("Unexpected end of file while reading module header")

        hdr = np.fromstring(hdr_bytes, dtype=VMPmodule_hdr, count=1)
        hdr_dict = dict(((n, hdr[n][0]) for n in VMPmodule_hdr.names))
        hdr_dict['offset'] = fileobj.tell()
        if read_module_data:
            hdr_dict['data'] = fileobj.read(hdr_dict['length'])
            if len(hdr_dict['data']) != hdr_dict['length']:
                raise IOError("""Unexpected end of file while reading data
                    current module: %s
                    length read: %d
                    length expected: %d""" % (hdr_dict['longname'],
                                              len(hdr_dict['data']),
                                              hdr_dict['length']))
            yield hdr_dict
        else:
            yield hdr_dict
            fileobj.seek(hdr_dict['offset'] + hdr_dict['length'], SEEK_SET)


class MPRFile():
    """Bio-Logic .mpr file

    The file format is not specified anywhere and has therefore been reverse
    engineered. Not all the fields are known.

    Attributes
    ==========
    modules - A list of dicts containing basic information about the
              'modules' of which the file is composed.

    data - numpy record array of type VMPdata_dtype containing the
           main data array of the file.

    startdate - The date when the experiment started

    enddate - The date when the experiment finished
    """

    def __init__(self, file_or_path):
        if isinstance(file_or_path, str):
            mpr_file = open(file_or_path, 'rb')
        else:
            mpr_file = file_or_path

        mpr_magic = b'BIO-LOGIC MODULAR FILE\x1a                         \x00\x00\x00\x00'  # noqa
        magic = mpr_file.read(len(mpr_magic))
        if magic != mpr_magic:
            raise ValueError('Invalid magic for .mpr file: %s' % magic)
        self.read_modules(mpr_file)
        mpr_file.close()

    def read_modules(self, mpr_file):

        modules = list(read_VMP_modules(mpr_file))
        self.modules = modules
        settings_mod, = (m for m in modules if m['shortname'] == b'VMP Set   ')
        data_module, = (m for m in modules if m['shortname'] == b'VMP data  ')
        maybe_log_module = [m for m in modules
                            if m['shortname'] == b'VMP LOG   ']

        with open('setting_data', 'wb') as f:
            f.write(maybe_log_module[0]['data'])

        n_data_points = np.fromstring(data_module['data'][:4], dtype='<u4')
        n_columns = np.fromstring(data_module['data'][4:5], dtype='u1')

        if data_module['version'] == 0:
            column_types = np.fromstring(data_module['data'][5:], dtype='u1',
                                         count=n_columns)
            remaining_headers = data_module['data'][5 + n_columns:100]
            main_data = data_module['data'][100:]
        elif data_module['version'] == 2:
            column_types = np.fromstring(data_module['data'][5:], dtype='<u2',
                                         count=n_columns)
            # There is 405 bytes of data before the main array starts
            remaining_headers = data_module['data'][5 + 2 * n_columns:405]
            main_data = data_module['data'][405:]
        else:
            raise ValueError("Unrecognised version for data module: %d" %
                             data_module['version'])

        if sys.version_info.major <= 2:
            assert(all((b == '\x00' for b in remaining_headers)))
        else:
            assert(not any(remaining_headers))

        results = VMPdata_dtype_from_colIDs(column_types)
        self.dtype, self.flags_dict, self.flags2_dict = results
        data = np.fromstring(main_data, dtype=self.dtype)
        assert(data.shape[0] == n_data_points)

        # Convert data to a pandas dataframe
        columns = self.dtype.names
        self.dataframe = pd.DataFrame(data, columns=columns)

        # No idea what these 'column types' mean or even if they are actually
        # column types at all
        self.version = int(data_module['version'])
        self.cols = column_types
        self.npts = n_data_points

        tm = time.strptime(str3(settings_mod['date']), '%m/%d/%y')
        self.startdate = date(tm.tm_year, tm.tm_mon, tm.tm_mday)

    def get_flag(self, flagname):
        if flagname in self.flags_dict:
            mask, dtype = self.flags_dict[flagname]
            return np.array(self.data['flags'] & mask, dtype=dtype)
        elif flagname in self.flags2_dict:
            mask, dtype = self.flags2_dict[flagname]
            return np.array(self.data['flags2'] & mask, dtype=dtype)
        else:
            raise AttributeError("Flag '%s' not present" % flagname)

    def active_mass(self):
        return 0
