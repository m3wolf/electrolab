import math

from units.predefined import define_units
import units

import default_units

define_units()

class CathodeLaminate():
    """Electrode material laminated onto a foil current collector. Units
    for active_material, carbon and binder are arbitrary, the ratio is
    more important.

    """
    def __init__(self, *, mass_active_material, mass_carbon, mass_binder, name):
        self.mass_active_material = mass_active_material
        self.mass_carbon = mass_carbon
        self.mass_binder = mass_binder
        self.name = name

    def active_ratio(self):
        """How much of the laminate material is electrochemically active. Does
        not account for substrate."""
        total_mass = (self.mass_active_material + self.mass_carbon + self.mass_binder)
        ratio = self.mass_active_material / total_mass
        return ratio


class CoinCellElectrode():
    """Goes in a coin-cell battery. Nominally made of a current collector
    substrate and a laminate of active material, carbon and
    binder. Mass should be specified using a unit from either the
    units package or `scimap.units`.

    - Mass: Mass of electrode plus substrate
    - Laminate: Laminated material on a substrate
    - substrate_mass: Estimated mass of the substrate by itself
    - diameter: Diameter of the punch electrode

    """
    def __init__(self, *, total_mass, substrate_mass, laminate, name, diameter):
        self.total_mass = total_mass
        self.substrate_mass = substrate_mass
        self.laminate = laminate
        self.diameter = diameter

    def area(self):
        return math.pi * (self.diameter/2)**2

    def mass_loading(self):
        """Calculate the mass per unit (geometric) area of active material."""
        active_mass = self.laminate.active_ratio() * (self.total_mass - self.substrate_mass)
        loading = active_mass / self.area()
        return default_units.electrode_loading(loading)
