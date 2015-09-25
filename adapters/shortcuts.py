# -*- coding: utf-8 -*-
import os

import adapters, exceptions

FILE_ADAPTERS = {
    '.plt': adapters.BrukerPltFile,
    '.xye': adapters.BrukerXyeFile,
    '.dat': adapters.FullProfDataFile,
    '.brml': adapters.BrukerBrmlFile
}

def adapter_from_filename(filename):
    """
    Use the given filename and extension to retrieve to appropriate
    file adapter.
    """
    # Determine file type from extension
    fileBase, extension = os.path.splitext(filename)
    # Prepare adapter for importing the file
    try:
        Adapter = FILE_ADAPTERS[extension]
    except KeyError:
        # Unknown file format, raise exception
        msg = 'Unknown file format {}.'.format(extension)
        raise exceptions.FileFormatError(msg)
    else:
        adapter = Adapter(filename)
        return adapter
