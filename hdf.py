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
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

import os

import h5py

import hdf
import exceptions
from utilities import prog

"""HDF is a format for storing large sets of data to disk. These
classes provide some wrappers around the h5py module.

The `Attr` data descriptor can be used to make HDF5 attributes seem as
though they are standard attributes.
"""


    # def __init__(self, default_scope="frameset"):
    #     self.default_scope = default_scope

# def hdf_attrs(Cls):
#     """Decorator that replaces the __getattr__, __setattr__, __delattr__
#     and __dir__ methods with ones that can retrieve values from an HDF
#     file. The decorated class should be given an `hdfattrs` attribute
#     that is dictionary of `Attribute` objects.
#     """
#     print(Cls)
#     if not hasattr(Cls, 'hdfattrs'):
#         msg = "Missing {}.hdfattrs dictionary.".format(Cls)
#         raise exceptions.HDFAttrsError(msg)
#     if not hasattr(Cls, 'hdf_default_scope'):
#         msg = "Missing {}.hdf_default_scope".format(Cls)
#         raise exceptions.HDFAttrsError(msg)
#     Cls.__getattr__ = get_hdf_attr
#     Cls.__setattr__ = set_hdf_attr
#     Cls.__delattr__ = del_hdf_attr
#     Cls.__dir__ = hdf_dir
#     return Cls

# def _get_group(obj, name, scope=None):
#     if scope is None:
#         scope = obj.hdf_default_scope
#     # Determine hdf group to use for attribute
#     if scope == 'subset':
#         group = obj.active_group
#     elif scope == 'frameset':
#         group = obj.frameset_group
#     elif scope == "frame":
#         group = obj.frame_group
#     else:
#         # Not a valid scope
#         msg = "Invalid attribute scope {scope} for {attr}"
#         msg = msg.format(scope=scope, attr=name)
#         raise exceptions.HDFScopeError(msg)
#     return group

# def get_hdf_attr(self, name):
#     if name == "hdf_default_scope":
#         value = "frameset"
#     # Raise an exception if the requested attribute is not listed as an HDF attr
#     elif not name in self.hdfattrs.keys():
#         msg = "'{cls}' object has no attribute '{attr}'"
#         raise AttributeError(
#             msg.format(cls=self.__class__.__name__, attr=name)
#         )
#     else:
#         # Get the attribute definition
#         attr = self.hdfattrs[name]
#         with self.hdf_file() as f:
#             group = _get_group(obj=self, scope=attr.scope, name=name)
#             # Retrieve the actual attribute value
#             try:
#                 value = f[group].attrs[attr.key]
#             except KeyError:
#                 value = attr.default
#     if attr.wrapper is not None:
#         value = attr.wrapper(value)
#     return value

# def set_hdf_attr(self, name, value):
#     if name in self.hdfattrs.keys():
#         # Set HDF attribute
#         attr = self.hdfattrs[name]
#         with self.hdf_file(mode="a") as f:
#             group = _get_group(obj=self, scope=attr.scope, name=name)
#             f[group].attrs[attr.key] = value
#     else:
#         # Set conventional python attribute
#         super(self.__class__, self).__setattr__(name, value)


# def del_hdf_attr(self, name):
#     if name in self.hdfattrs.keys():
#         # Set HDF attribute
#         attr = self.hdfattrs[name]
#         with h5py.File(self.hdf_file, mode='a') as f:
#             group = _get_group(obj=self, scope=attr.scope, name=name)
#             del f[group].attrs[name]
#     else:
#         # Set conventional python attribute
#         super(self.__class__, self).__delattr__(name)


# def hdf_dir(self):
#     # Get regular list of attributes
#     attrs = super(self.__class__, self).__dir__()
#     # Add HDF attributes
#     attrs += self.hdfattrs.keys()
#     return attrs


# class Attr():
#     """A class that describes a value that should be stored to disk as an
#     HDF5 attribute. A class should have a list of these named
#     _hdfattrs and use getattr, setattr, and delattr defined in this
#     module to override accessors.

#     """
#     def __init__(self, key, default=None, wrapper=None, scope=None):
#         self.key = key
#         self.default = default
#         self.wrapper = wrapper
#         self.scope = scope

#     def __str__(self):
#         return "<Attr: {} ({})>".format(key, scope)


def prepare_hdf_group(filename: str, groupname: str, dirname: str):
    """Check the filenames and create an hdf file as needed. Will
    overwrite the group if it already exists.

    Returns: HDFFile

    Arguments
    ---------

    - filename : name of the requested hdf file, may be None if not
      provided, in which case the filename will be generated
      automatically based on `dirname`.

    - groupname : Requested groupname for these data.

    - dirname : Used to derive a default filename if None is passed
      for `filename` attribute.
    """
    # Get default filename and groupname if necessary
    if filename is None:
        real_name = os.path.abspath(dirname)
        new_filename = os.path.split(real_name)[1]
        hdf_filename = "{basename}-results.h5".format(basename=new_filename)
    else:
        hdf_filename = filename
    if groupname is None:
        groupname = os.path.split(os.path.abspath(dirname))[1]
    # Open actual file
    hdf_file = h5py.File(hdf_filename)
    # Alert the user that we're overwriting this group
    if groupname in hdf_file.keys():
        if not prog.quiet:
            msg = 'Group "{groupname}" exists. Overwriting.'
            print(msg.format(groupname=groupname))
        del hdf_file[groupname]
    new_group = hdf_file.create_group(groupname)
    # User feedback
    if not prog.quiet:
        print('Saving to HDF5 file {file} in group {group}'.format(
            file=hdf_filename,
            group=groupname)
        )
    return new_group


class Attr():
    """A data descriptor that retrieves a value that should be stored to
    disk as an HDF5 attribute.

    Arguments
    ---------
    - key : What to use as the HDF5 attr name

    - default : Value to use if the HDF5 group does not have this attribute set

    - wrapper : Optional function to call when retrieving a
      value. Useful for retrieving data that cannot be stored directly
      as data (eg namedtuple)

    - scope : Where to look for the attribute. If omitted or None, the
      objects value for `hdf_default_scope` will be used. Options are:
      - "frameset": Top-level HDF5 group
      - "subset": Iteration of this frameset group eg. aligned frames
      - "frame": The specific frame that this descriptor is attached to.

    """
    def __init__(self, key: str, default=None, wrapper=None, scope=None):
        self.key = key
        self.default = default
        self.wrapper = wrapper
        self.scope = scope

    def __get__(self, obj, cls):
        # Get the attribute definition
        with obj.hdf_file() as f:
            group = self._get_group(obj=obj)
            # Retrieve the actual attribute value
            try:
                value = f[group].attrs[self.key]
            except KeyError:
                value = self.default
        # Execute the wrapper function if it was provided at load-time
        if self.wrapper is not None:
            value = self.wrapper(value)
        return value

    def __set__(self, obj, val):
        # Set HDF attribute
        with obj.hdf_file(mode="a") as f:
            group = self._get_group(obj=obj)
            f[group].attrs[self.key] = val

    def __delete__(self, obj):
        with obj.hdf_file(mode="a") as f:
            group = self._get_group(obj=obj)
            del f[group].attrs[self.key]

    def _get_group(self, obj):
        """Retrieve the appropriate HDF group given this descriptors scope."""
        if self.scope is None:
            scope = obj.hdf_default_scope
        else:
            scope = self.scope
        # Determine hdf group to use for attribute
        if scope == 'subset':
            group = obj.active_group
        elif scope == 'frameset':
            group = obj.frameset_group
        elif scope == "frame":
            group = obj.frame_group
        else:
            # Not a valid scope
            msg = "Invalid attribute scope {scope} for {attr}"
            msg = msg.format(scope=scope, attr=self.key)
            raise exceptions.HDFScopeError(msg)
        return group
