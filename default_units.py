# -*- coding: utf-8 -*-
"""
Define common units across the whole application. Dataframes
assume the following units and can convert after calculation.
"""

import units, units.predefined

# Define default units
units.predefined.define_units()
# Grams
mass = units.unit('g')
# Time (hours)
time = units.unit('h')
# Milli-amp hours
capacity = units.unit('mA') * units.unit('h')
# Milli-amp hours per gram
specific_capacity = units.unit('mA') * units.unit('h') / units.unit('g')
# Volts
potential = units.unit('V')
# Electrode mass loading
cm = units.unit('cm')
electrode_loading = units.unit('mg')/(cm*cm)
