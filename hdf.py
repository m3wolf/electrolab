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

import h5py

import exceptions

"""HDF is a format for storing large sets of data to disk. These
classes provide some wrappers around the h5py module.

The `Attr` class combined with `getattr`, `setattr` and `delattr`
functions can be used to make HDF5 attributes seem as though they are
standard attributes.

"""



    # def __init__(self, default_scope="frameset"):
    #     self.default_scope = default_scope

def hdf_attrs(Cls):
    """Decorator that replaces the __getattr__, __setattr__, __delattr__
    and __dir__ methods with ones that can retrieve values from an HDF
    file. The decorated class should be given an `hdfattrs` attribute
    that is dictionary of `Attribute` objects.
    """
    if not hasattr(Cls, 'hdfattrs'):
        msg = "Missing {}.hdfattrs dictionary.".format(Cls)
        raise exceptions.HDFAttrsError(msg)
    if not hasattr(Cls, 'hdf_default_scope'):
        msg = "Missing {}.hdf_default_scope".format(Cls)
        raise exceptions.HDFAttrsError(msg)
    Cls.__getattr__ = get_hdf_attr
    Cls.__setattr__ = set_hdf_attr
    Cls.__delattr__ = del_hdf_attr
    Cls.__dir__ = hdf_dir
    return Cls

def _get_group(obj, name, scope=None):
    if scope is None:
        scope = obj.hdf_default_scope
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
        msg = msg.format(scope=scope, attr=name)
        raise exceptions.HDFScopeError(msg)
    return group

def get_hdf_attr(self, name):
    if name == "hdf_default_scope":
        value = "frameset"
    # Raise an exception if the requested attribute is not listed as an HDF attr
    elif not name in self.hdfattrs.keys():
        msg = "'{cls}' object has no attribute '{attr}'"
        raise AttributeError(
            msg.format(cls=self.__class__.__name__, attr=name)
        )
    else:
        # Get the attribute definition
        attr = self.hdfattrs[name]
        with self.hdf_file() as f:
            group = _get_group(obj=self, scope=attr.scope, name=name)
            # Retrieve the actual attribute value
            try:
                value = f[group].attrs[attr.key]
            except KeyError:
                value = attr.default
    if attr.wrapper is not None:
        value = attr.wrapper(value)
    return value

def set_hdf_attr(self, name, value):
    if name in self.hdfattrs.keys():
        # Set HDF attribute
        attr = self.hdfattrs[name]
        with self.hdf_file(mode="a") as f:
            group = _get_group(obj=self, scope=attr.scope, name=name)
            f[group].attrs[name] = value
    else:
        # Set conventional python attribute
        super(self.__class__, self).__setattr__(name, value)


def del_hdf_attr(self, name):
    if name in self.hdfattrs.keys():
        # Set HDF attribute
        attr = self.hdfattrs[name]
        with h5py.File(self.hdf_file, mode='a') as f:
            group = _get_group(obj=self, scope=attr.scope, name=name)
            del f[group].attrs[name]
    else:
        # Set conventional python attribute
        super(self.__class__, self).__delattr__(name)


def hdf_dir(self):
    # Get regular list of attributes
    attrs = super(self.__class__, self).__dir__()
    # Add HDF attributes
    attrs += self.hdfattrs.keys()
    return attrs


class Attr():
    """A class that describes a value that should be stored to disk as an
    HDF5 attribute. A class should have a list of these named
    _hdfattrs and use getattr, setattr, and delattr defined in this
    module to override accessors.

    """
    def __init__(self, key, default=None, wrapper=None, scope=None):
        self.key = key
        self.default = default
        self.wrapper = wrapper
        self.scope = scope

    def __str__(self):
        return "<Attr: {} ({})>".format(key, scope)

    # @property
    # def scope(self):
    #     if self._scope is None:
    #         # Default frameset type
    #         return "frameset"
    #     else:
    #         return self._scope


class HDFAttribute():
    """A descriptor that returns an HDF5 attribute if possible or else an
    in-memory value. An optional `wrapper` argument will wrap the data
    in this function before returning it. Assumes the object to which
    it is attached has an attribute/property `hdf_node` that returns a
    node object.

    Arguments
    ---------
    attribute_name (string): name to be assigned to the HDF5 attribute
        storing this datum

    default (any): Default value to return if no value is stored

    wrapper (callable): Returned values get sent through this callable first

    group_func (string): Name of object attribute to call in order to
        retrieve the hdf5 group that this attr should be saved
        to. Default 'hdf_node'.
    """
    def __init__(self, attribute_name, default=None, wrapper=None,
                 group_func='hdf_node'):
        self.attribute_name = attribute_name
        self.default_value = default
        self.wrapper = wrapper
        self.group_func = group_func

    def _attrs(self, obj):
        # Retrieve the HDF5 group
        getter = getattr(obj, self.group_func)
        # Retrieve the attrs dictionary
        attrs = getattr(getter(), 'attrs', obj._attrs)
        return attrs

    def __get__(self, obj, owner):
        attrs = self._attrs(obj)
        value = attrs.get(self.attribute_name, self.default_value)
        if self.wrapper:
            value = self.wrapper(value)
        return value

    def __set__(self, obj, value):
        attrs = self._attrs(obj)
        attrs[self.attribute_name] = value

    def __delete__(self, obj):
        attrs = self._attrs(obj)
        del attrs[self.attribute_name]
