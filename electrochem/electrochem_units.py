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

import units

# Prepare some units specific to electrochemistry
hour = units.unit('h')
mAh = units.named_unit('mAh', ['mA', 'h'], [])
uAh = units.scaled_unit('µAh', 'mAh', 10**-3)
units.named_unit('µA', ['uA'], [])
