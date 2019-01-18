# -*- coding: utf-8 -*-

# This file is part of Foobar.
# 
# Foobar is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Foobar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

"""A set of tools for calculating X-ray behavior and properties of
materials.

"""

import re
from functools import reduce

from . import exceptions


def parse_chemical_formula(formula: str) -> list:
    """Parse a chemical formula into it's elemental components.
    
    For example, passing in "H2O" will produce the followign output:
    
      [('H', 2), ('O', 1)]
    
    Parameters
    ==========
    formula
      A chemical formula, eg. LiMn2O4.
    
    Returns
    =======
    elements
      A list of tuples. Each tuple is an element with (sym, num)
      order.
    
    """
    # Match individual element components with a regular expression
    regex = re.compile('([A-Z][a-z]?)[_{]*([.0-9]*)')
    match = regex.findall(formula)
    if not match:
        raise exceptions.ChemicalFormulaError(
            "Could not parse chemical formula: {}"
            "".format(formula))
    # Parse the match strings and turn into integers
    elements = []
    for (elem, num) in match:
        if num == '':
            num = 1
        else:
            num = float(num)
        elements.append((elem, num))
    return elements


def molar_mass(formula: str) -> float:
    """Calculate the molar mass of a chemical formula.
    
    Parameters
    ==========
    formula
      A chemical formula, eg. LiMn2O4.
    
    Returns
    =======
    molar_mass
      The molar mass, in g/mol.
    
    """
    elements = parse_chemical_formula(formula)
    molar_mass = 0
    for elem, num in elements:
        molar_mass += num * element_masses[elem]
    return molar_mass


def mass_attenuation_coefficient(formula: str, xray_energy: float) -> float:
    """Calculate attenuation for a given chemical formula.
    
    Parameters
    ==========
    formula
      Chemical formula, eg. LiMn2O4.
    xray_energy
      The X-ray energy, in electron-volts.
    
    Returns
    =======
    coefficient
      The calculated mass attenuation coefficient, in cm²g⁻¹
    
    """
    total = 0
    # For each element, add its *atomic* coefficient
    elements = parse_chemical_formula(formula)
    for elem, num in elements:
        elem_molar_mass = molar_mass(elem)
        cross_section = photoabsorption_cross_section(elem, xray_energy)
        total += num * cross_section * elem_molar_mass
    # Convert from atomic coefficient to mass coefficient
    molar_mass_ = molar_mass(formula)
    coefficient = total / molar_mass_
    return coefficient


def photoabsorption_cross_section(element: str, xray_energy: float) -> float:
    """Provide estimated photoabsorption cross section for an element.
    
    Parameters
    ==========
    element
      Symbol for desired element, eg. 'Li'.
    xray_energy
      The X-ray energy, in electron-volts.
    
    Returns
    =======
    cross_section
      The reported absorption cross section, in cm²g⁻¹
    
    """
    return _abs_cross_sections[element][xray_energy]


# This is a kludge until full data files can be retrieved from NIST
# after government shutdown ends (fuck you, Trump)
_abs_cross_sections = {
    'H': {
        8047.8: 5.7636e-03,
    },
    'Li': {
        8047.8: 0.2514,
    },
    'C': {
        8047.8: 4.179,
    },
    'O': {
        8047.8: 11.02,
    },
    'F': {
        8047.8: 15.13,
    },
    'Al': {
        8047.8: 46.81,
    },
    'Mn': {
        8047.8: 269.6,
    },
    'Co': {
        8047.8: 316.1,
    },
    'Ni': {
        8047.8: 46.69,
    },
}


element_masses = {
    'H': 1.0079,
    'Li': 6.941,
    'C': 12.01,
    'O': 16.00,
    'F': 19.00,
    'Mn': 54.96,
    'Al': 26.98,
    'Co': 58.93,
    'Ni': 58.69,
}
