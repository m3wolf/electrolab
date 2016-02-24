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

"""A collection of classes and functions that arent' specific to any
one type of measurement. Also, it defines some namedtuples for
describing coordinates.
"""

from collections import namedtuple
import sys

from tqdm import tqdm

xycoord = namedtuple('xycoord', ('x', 'y'))
Pixel = namedtuple('Pixel', ('vertical', 'horizontal'))
shape = namedtuple('shape', ('rows', 'columns'))


class Prog:
    """A progress bar for displaying how many iterations have been
    completed. This is mostly just a wrapper around the tqdm
    library. Additionally it makes use of the borg pattern, so setting
    Prog.quiet to True once silences all progress bars. This is useful
    for testing.
    """
    __global_state = {
        'quiet': False
    }

    def __init__(self):
        self.__dict__ = self.__global_state

    def __call__(self, iterable, *args, **kwargs):
        """Progress meter. Wraps around tqdm with some custom defaults."""
        if self.quiet:
            # Supress outputs by redirecting to stdout
            # kwargs['file'] = open(os.devnull, 'w')
            # # kwargs['file'] = os.devnull
            ret = iterable
        else:
            kwargs['file'] = kwargs.get('file', sys.stdout)
            kwargs['leave'] = kwargs.get('leave', True)
            ret = tqdm(iterable, *args, **kwargs)
        return ret

prog = Prog()
