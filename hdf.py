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

"""HDF is a format for storing large sets of data to disk. These
classes provide some wrappers around the h5py module."""


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
