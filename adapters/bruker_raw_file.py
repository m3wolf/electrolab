# # -*- coding: utf-8 -*-
from collections import namedtuple
import codecs

import exceptions

ByteField = namedtuple('ByteField', ('name', 'type', 'length'))

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


class BrukerRawFile():

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
                    ENCODING
                )
                currentPos += field.length
            result = FieldValue(name=field.name, value=value)
            yield(result)

    def parse_header(self):
        self.header = {}
        for field in self.byte_fields(HEADER_BYTES):
            self.header[field.name] = field.value

    def extract_string(self, offset):
        marker = offset
        # Chop bytestring up by looking for a newline
        while chr(self.data[marker]) != '\n':
            marker += 1
        byteString = self.data[offset:marker]
        newString = codecs.decode(byteString, ENCODING)
        return newString, marker - offset
