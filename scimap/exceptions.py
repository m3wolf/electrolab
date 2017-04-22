# -*- coding: utf-8 -*-
#
# Copyright © 2016 Mark Wolf
#
# This file is part of Scimap.
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

"""Define some classes for more fine-grained control over exception
handling."""

#############################
# Electrochemistry exceptions
#############################
class ReadCurrentError(Exception):
    """Cannot read a current from electrochemistry cycling file."""
    pass

##############################
# X-Ray Diffraction exceptions
##############################
class UnitCellError(ValueError):
    """The given unit-cell parameters are invalid for the crystal
    system.
    """
    pass

class HKLFormatError(ValueError):
    """The given hkl index is vague of malformed."""
    pass

class RefinementError(Exception):
    """Generic error: we tried to refine something but it didn't work."""
    pass

class PeakFitError(RefinementError):
    """Tried to fit curves to a peak but could not find a local minimum."""
    pass

class EmptyRefinementError(RefinementError):
    """There were no parameters to refine."""
    pass

class SingularMatrixError(RefinementError):
    """FullProf refinement reached an invalid Hessian and was not able to
    continue."""
    def __init__(self, param):
        self.param = param

    def __str__(self):
        msg = "Singular matrix while refining {param}".format(param=self.param)
        return msg

class DivergenceError(RefinementError):
    """Fullprof refinement did not reach a local maximum."""
    pass

class PCRFileError(RefinementError):
    """Tried to read a Fullprof PCR file but it did not conform to the
    expected format."""
    pass

class NoReflectionsError(RefinementError):
    """The refinement has I(obs) = 0. (Do you really have reflections?)"""
    def __str__(self):
        msg = "I(obs) = 0. Do you really have reflections? {}"
        return msg.format(self.args)

class UnknownFileTypeError(ValueError):
    """This file does not have a handler registered."""
    pass

class FileFormatError(ValueError):
    """This file is not formatted as expected."""
    pass

class DataNotFoundError(FileNotFoundError):
    """Expected a directory containing data but found none."""
    pass

class FrameFileNotFound(IOError):
    """Expected to load a TXM frame file but it doesn't exist."""
    pass

class GroupKeyError(KeyError):
    """Tried to load an HDF group but the group doesn't exist or is
    ambiguous."""
    pass

class DataFormatError(RuntimeError):
    """The raw data are arranged in a way that the importers or TXM classes do
    not understand.
    """
    pass

class HDFScopeError(ValueError):
    """Tried to pass an HDF scope that is not recognized."""
    pass

class HDFAttrsError(ValueError):
    """Class uses hdf_attrs decorator but dosn't have hdfattrs attribute."""
    pass

class FileExistsError(IOError):
    """Tried to import a TXM frameset but the corresponding HDF file
    already exists."""
    pass

class CreateGroupError(ValueError):
    """Tried to import a TXM frameset into a group but the corresponding
    HDF group already exists.
    """
    pass

class FilenameParseError(ValueError):
    """The parameters in the filename do not match the naming scheme
    associated with this flavor."""
    pass

class DatasetExistsError(RuntimeError):
    """Trying to save a new dataset but one already exists with the given
    path."""
    pass

class NoParticleError(Exception):
    pass
