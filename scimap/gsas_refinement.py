# -*- coding: utf-8 -*-
#
# Copyright © 2018 Mark Wolf
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

# flake8: noqa


from .base_refinement import BaseRefinement
import tempfile
import sys
import os
import logging

log = logging.getLogger(__name__)

# Import the GSAS-II modules
GSAS_LOCATIONS = (
    '~/build/pyGSAS',
    '~/g2conda/GSASII',
    '~/src/pyGSAS',
)

def import_gsas(locations):
    """Look for valid GSAS-II install and import as ``gsas``."""
    for path in locations:
        path = os.path.expanduser(path)
        # Existant path found, so try importing
        if os.path.exists(path):
            sys.path.insert(0, path)
            try:
                import GSASIIscriptable as gsas
            except ImportError:
                log.debug("GSAS-II directory not importable: %s", path)
                pass
            else:
                log.debug("Found GSAS-II directory: %s", path)
                break
            finally:
                sys.path.pop(0)
    # Verify that GSAS was imported
    if 'gsas' not in locals().keys():
        raise ImportError("GSAS-II not found in any of %s. "
                          "Make sure it is in your PYTHON_PATH "
                          "or add install location to "
                          "GSAS_LOCATIONS in %s."
                          "" % (str(locations), __file__))
    return gsas
gsas = import_gsas(GSAS_LOCATIONS)    


class GSASRefinement(BaseRefinement):
    _gpx = None

    def __init__(self, gpx_template=None, *args, **kwargs):
        self.gpx_template = gpx_template
        super().__init__(*args, **kwargs)
   
    @property
    def gpx(self):
        """Create a GSAS-II GPX project file.
        
        This file is necessary for further GSAS refinement operations.
        
        """
        if self._gpx is None:
            # Check if an existing file should be used
            gpx_fname = self.file_root + '.gpx'
            if os.path.exists(gpx_fname):
                gpx_kw = dict(gpxfile=gpx_fname)
            else:
                gpx_kw = dict(newgpx=gpx_fname)
            # Create the GPX project file
            self._gpx = gsas.G2Project(**gpx_kw)
            # Make any necessary parent directories
            dir_, fname = os.path.split(self._gpx.filename)
            if not os.path.exists(dir_):
                os.makedirs(dir_)
            # Save the newly created GPX file
            self._gpx.save()
        return self._gpx

    def refine_all(self):
        pass
